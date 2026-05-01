import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { SystemMessage, AIMessage } from "@langchain/core/messages";
import { loadModel } from "../../../utils/agents/loadModel.js";
import { flightTools } from "./tools.js";
import { summarizeFlights } from "./utils/summarizeFlights.js";
import { extractLastToolJson } from "../../../utils/agents/extractLastToolJson.js";
import { generator } from "../../../utils/agents/generator.js";
import { nanoid } from "nanoid";
import type {
  FlightResults,
  FlightLeg,
  AirportInfo,
} from "../../../types/flight/flights.js";
import type { AgentStateType } from "../../state.js";
import type { Trip } from "../../../types/trip.js";

const useFlightApi = process.env.USE_FLIGHT_API === "true";
const GENERATE_SUMMARIES = process.env.GENERATE_SUMMARIES === "true";

const model = loadModel("fast");

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

function createFlightTemplate(): FlightResults {
  return {
    id: nanoid(),
    price: null as unknown as number,
    currency: null as unknown as string,
    legs: null as unknown as FlightLeg[],
    destinationAirport: null as unknown as AirportInfo,
    destinationCity: null,
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
    const flights = await generator<FlightResults>({
      data: Array.from({ length: 5 }, () => createFlightTemplate()),
      context: buildTripContext(trip),
      description:
        "round-trip flight options. Each flight must have exactly 2 legs: one outbound (origin to destination) and one return (destination to origin). Each leg needs a direction ('outbound' or 'return'), legDuration, and a segments array. Each segment needs duration, departure (airport IATA code + ISO time), arrival (airport IATA code + ISO time), and airline name. Prices should be realistic USD values.",
    });

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
