import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { SystemMessage, HumanMessage, AIMessage } from "@langchain/core/messages";
import { loadModel } from "../../../utils/agents/loadModel.js";
import { flightTools } from "./tools.js";
import { summarizeFlights } from "./utils/summarizeFlights.js";
import { extractLastToolJson } from "../../../utils/agents/extractLastToolJson.js";
import { nanoid } from "nanoid";
import * as z from "zod";
import type { FlightResults } from "../../../types/flight/flights.js";
import type { AgentStateType } from "../../state.js";
import type { Trip } from "../../../types/trip.js";

const useFlightApi = process.env.USE_FLIGHT_API === "true";
const GENERATE_SUMMARIES = process.env.GENERATE_SUMMARIES === "true";

const model = loadModel("fast");
// "standard" tier (e.g. gpt-4o-mini) is enough here — withStructuredOutput
// enforces shape, so we trade model intelligence for latency.
const flightGenModel = loadModel("standard");

// Mirrors the api-gateway's FlightSchema. Used by the LLM-generator path to
// constrain the model output via withStructuredOutput, which is enforced at
// the provider API level (no prompt-engineering drift).
const flightOutputSchema = z.object({
  flights: z.array(
    z.object({
      price: z.string().describe("Total round-trip price as a string, e.g. '423.50'"),
      currency: z.string().describe("Currency code, e.g. 'USD'"),
      legs: z.array(
        z.object({
          direction: z.enum(["outbound", "return"]),
          legDuration: z.string().describe("ISO 8601 duration string, e.g. 'PT5H30M'"),
          segments: z.array(
            z.object({
              duration: z.string().describe("ISO 8601 duration string"),
              departure: z.object({
                airport: z.string().describe("3-letter IATA code"),
                time: z.string().describe("ISO 8601 datetime string"),
              }),
              arrival: z.object({
                airport: z.string().describe("3-letter IATA code"),
                time: z.string().describe("ISO 8601 datetime string"),
              }),
              airline: z.string(),
            }),
          ),
        }),
      ),
      destinationAirport: z.object({
        name: z.string(),
        iata_code: z.string().describe("3-letter IATA code"),
        latitude_deg: z.number(),
        longitude_deg: z.number(),
      }),
      destinationCity: z
        .object({
          name: z.string(),
          latitude: z.number(),
          longitude: z.number(),
        })
        .nullable(),
    }),
  ),
});

function getMissingFields(trip: Trip): string[] {
  const missing: string[] = [];
  if (!trip.origin) missing.push("origin city/airport");
  if (!trip.destination) missing.push("destination city/airport");
  if (!trip.departureDate) missing.push("departure date");
  if (!trip.returnDate) missing.push("return date");
  return missing;
}

function buildTripContext(trip: Trip): Record<string, unknown> {
  return {
    origin: trip.origin,
    destination: trip.destination,
    departureDate: trip.departureDate,
    returnDate: trip.returnDate,
    budget: trip.budget,
    interests: trip.interests,
    constraints: trip.constraints,
  };
}

function buildSystemPrompt(trip: Trip): string {
  const missingFields = getMissingFields(trip);
  const tripContext = `
Current trip details:
- Origin: ${trip.origin || "not specified"}
- Destination: ${trip.destination || "not specified"}
- City: ${trip.city || "not specified"}
- Departure date: ${trip.departureDate || "not specified"}
- Return date: ${trip.returnDate || "not specified"}
- Budget: ${trip.budget ? `$${trip.budget}` : "not specified"}

${missingFields.length > 0 ? `Missing required information: ${missingFields.join(", ")}` : "All required flight information is available."}
`;

  return `
You are a helpful flight research assistant helping plan a trip.

${tripContext}

You can search for round-trip flights using the tools available to you.
Each tool has specific required parameters - review them carefully.

Rules:
- If required trip information is missing, ask for it ONE piece at a time.
- Prioritize missing fields in this order: origin, destination, departure date, return date.
- When you have all required info, search for flights.
- You MUST use tools to get flight data.
- You MUST NOT invent or guess flight information.
- Be CONCISE. Keep responses short (1-2 sentences max).
- Do NOT format or present detailed results to the user (that's handled separately).
- If the user provides an airline, convert the airline name to its 2-character IATA code.
- When calling searchFlights, pass cityName if the destination city name is known (use the City field from trip details above).
`;
}

/**
 * Flight agent node — uses Amadeus API (USE_FLIGHT_API=true) or
 * LLM-generated data via the generator utility (USE_FLIGHT_API=false).
 */
export async function flightNode(
  state: AgentStateType,
): Promise<Partial<AgentStateType>> {
  if (useFlightApi) {
    return flightNodeWithApi(state);
  }
  return flightNodeWithGenerator(state);
}

/**
 * API path: uses createReactAgent with the searchFlights tool (Amadeus API).
 */
async function flightNodeWithApi(
  state: AgentStateType,
): Promise<Partial<AgentStateType>> {
  const inputMessageCount = state.messages.length;
  let currentMessages = state.messages;
  const trip = state.trip;

  try {
    const flightAgent = createReactAgent({
      llm: model,
      tools: flightTools,
      messageModifier: new SystemMessage(buildSystemPrompt(trip)),
    });

    const result = await flightAgent.invoke({ messages: state.messages });
    currentMessages = result.messages;

    // Check if tools were called THIS turn by looking at new messages only
    const newMessages = result.messages.slice(inputMessageCount);
    const toolsCalledThisTurn = newMessages.some((m) => m.type === "tool");

    // If no tools were called, the agent is asking for clarification
    if (!toolsCalledThisTurn) {
      return { messages: result.messages };
    }

    // Post-process flight results
    const flightData = extractLastToolJson<FlightResults[]>(result.messages);

    // Detect tool-returned error (e.g., from validateAirportCode)
    if (
      flightData &&
      !Array.isArray(flightData) &&
      (flightData as unknown as { error: boolean }).error === true
    ) {
      const errorMsg =
        (flightData as unknown as { message: string }).message ??
        "Something went wrong.";

      const errorMessage = new AIMessage(errorMsg);

      return {
        messages: [...result.messages, errorMessage],
        data: {
          type: "error",
          message: errorMsg,
        },
      };
    }

    // Validate flight data
    if (!Array.isArray(flightData) || flightData.length === 0) {
      console.error(
        "[flightNode] Flight data is empty or not an array:",
        flightData,
      );
      const errorMessage = new AIMessage(
        "Something went wrong. Please try again later.",
      );
      return { messages: [...result.messages, errorMessage] };
    }

    // Summarize the flights - non-fatal if LLM call fails
    let summary = "";
    if (GENERATE_SUMMARIES) {
      try {
        summary = await summarizeFlights(flightData, result.messages);
      } catch (summarizeError) {
        console.error("[flightNode] Summarization failed:", summarizeError);
        summary = "Here are the flight options I found.";
      }
    }

    const finalMessage = new AIMessage(
      summary || "Here are the flight options I found.",
    );

    // Return updated state with data extracted
    return {
      messages: [...result.messages, finalMessage],
      data: {
        type: "flight",
        summary,
        options: flightData,
      },
    };
  } catch (error) {
    console.error("[flightNode] Post-processing error:", error);
    const errorMessage = new AIMessage(
      "Something went wrong. Please try again later.",
    );
    return { messages: [...currentMessages, errorMessage] };
  }
}

/**
 * Generator path: uses the LLM generator utility to produce flight data.
 */
async function flightNodeWithGenerator(
  state: AgentStateType,
): Promise<Partial<AgentStateType>> {
  const trip = state.trip;
  const missingFields = getMissingFields(trip);

  // If required fields are missing, ask for them
  if (missingFields.length > 0) {
    const response = await model.invoke([
      new SystemMessage(`You are a helpful flight research assistant.
You need the user's trip details to recommend flights.
Ask for the missing information ONE piece at a time, in priority order.
Be concise (1-2 sentences max).
Missing: ${missingFields.join(", ")}`),
      ...state.messages.slice(-6),
    ]);

    const aiMessage = new AIMessage(response.content as string);
    return { messages: [...state.messages, aiMessage] };
  }

  try {
    const structuredModel = flightGenModel.withStructuredOutput(flightOutputSchema);
    const tripContext = JSON.stringify(buildTripContext(trip), null, 2);

    const { flights: generatedFlights } = await structuredModel.invoke([
      new SystemMessage(`Generate 3 plausible round-trip flight options for the trip below.

Rules:
- Output must conform to the provided schema exactly.
- Each flight has exactly 2 legs: one outbound, one return.
- Use realistic flight times, durations, prices, and major airlines.
- Provide approximate latitude/longitude (decimal degrees) for the destination airport and city.`),
      new HumanMessage(`Trip details:\n${tripContext}`),
    ]);

    const flights = generatedFlights.map(
      (f) => ({ id: nanoid(), ...f }) as unknown as FlightResults,
    );

    const summary = GENERATE_SUMMARIES
      ? await summarizeFlights(flights, state.messages)
      : "";
    const aiMessage = new AIMessage(summary || "Here are your flight options.");

    return {
      messages: [...state.messages, aiMessage],
      data: {
        type: "flight",
        summary,
        options: flights,
      },
    };
  } catch (error) {
    console.error("[flightNode] Generator error:", error);
    const errorMessage = new AIMessage(
      "Something went wrong finding flights. Please try again later.",
    );
    return { messages: [...state.messages, errorMessage] };
  }
}
