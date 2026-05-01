import { tool } from "@langchain/core/tools";
import * as z from "zod";
import type {
  FlightResults,
  FlightLeg,
  FlightSegment,
  AirportInfo,
  CityInfo,
} from "../../types/flight/flights.js";
import { getAmadeusToken } from "../../utils/amadeus/tokenManager.js";
import { validateAirportCode } from "./validateAirport.js";
import {
  searchWikipediaPageId,
  fetchWikipediaCoordinates,
} from "./searchWikipedia.js";
import { nanoid } from "nanoid";

/**
 * IMPORTANT:
 * - Tool returns structured data
 * - Tool NEVER formats for humans
 * - Tool NEVER reasons
 */

export const searchFlights = tool(
  async ({
    originLocationCode,
    destinationLocationCode,
    departureDate,
    returnDate,
    includedAirlinesCodes,
    cityName,
  }) => {
    let destinationAirportInfo: AirportInfo;
    try {
      await validateAirportCode(originLocationCode);
      destinationAirportInfo = await validateAirportCode(
        destinationLocationCode,
      );
    } catch {
      return JSON.stringify({ error: true, message: "Invalid Airport." });
    }

    // Fetch city-centre coordinates from Wikipedia when city name is provided
    let destinationCityInfo: CityInfo | null = null;
    if (cityName) {
      const pageId = await searchWikipediaPageId(cityName);

      if (pageId !== null) {
        const coords = await fetchWikipediaCoordinates(pageId);
        if (coords) {
          destinationCityInfo = {
            name: cityName,
            latitude: coords.latitude,
            longitude: coords.longitude,
          };
        }
      }
    }

    const token = await getAmadeusToken();

    const url = new URL(
      "https://test.api.amadeus.com/v2/shopping/flight-offers",
    );
    url.searchParams.set("originLocationCode", originLocationCode);
    url.searchParams.set("destinationLocationCode", destinationLocationCode);
    url.searchParams.set("departureDate", departureDate);
    url.searchParams.set("returnDate", returnDate);
    url.searchParams.set("max", String(5)); // Set a default of 5 results
    url.searchParams.set("currencyCode", "USD");
    url.searchParams.set("adults", String(1));
    url.searchParams.set("excludedAirlineCodes", "6X"); // Removes the fake Amadeus Airline (AMADEUS SIX - 6X) that they include in the data

    // Only include airline codes if provided
    if (includedAirlinesCodes && includedAirlinesCodes.length > 0) {
      url.searchParams.set(
        "includedAirlinesCodes",
        includedAirlinesCodes.join(","),
      );
    }

    // console.log(url.toString());

    const response = await fetch(url.toString(), {
      method: "GET",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
    });

    // if (!response.ok) {
    //   const text = await response.text();
    //   throw new Error(
    //     `Flight API error: ${response.status} ${response.statusText} ${text}`
    //   );
    // }

    if (!response.ok) {
      return JSON.stringify({ error: true, message: "Flight API unavailable" });
    }

    const rawData = await response.json();
    const dictionary = rawData.dictionaries; // Contains the mapping between codes and real names
    const carrierMap = dictionary?.carriers ?? [];

    // Shape the results
    const allFlightResults: FlightResults[] = [];

    // Loop through the offers
    for (const offer of rawData.data) {
      const flightResults: FlightResults = {
        id: nanoid(),
        price: 0,
        currency: "",
        legs: [],
        destinationAirport: destinationAirportInfo,
        destinationCity: destinationCityInfo,
      };

      // const duration = offer.duration;
      const currency = offer.price.currency;
      const price = offer.price.total;

      // flightResults.duration = duration;
      flightResults.price = price;
      flightResults.currency = currency;

      // Get outbound itinerary
      const outboundItinerary = offer.itineraries[0];

      // Outbound Itinerary
      const outboundLeg: FlightLeg = {
        direction: "outbound",
        legDuration: outboundItinerary.duration,
        segments: [],
      };
      // Loop through the segments of the leg
      for (const flight of outboundItinerary.segments) {
        const duration = flight.duration;
        const departure = flight.departure;
        const arrival = flight.arrival;
        const carrierCode = flight.carrierCode;

        const airlineName = carrierMap[carrierCode] ?? carrierCode;

        const flightSegment: FlightSegment = {
          duration: duration,
          departure: {
            airport: departure.iataCode,
            time: departure.at,
          },
          arrival: {
            airport: arrival.iataCode,
            time: arrival.at,
          },
          airline: airlineName,
        };

        outboundLeg.segments.push(flightSegment);
      }
      flightResults.legs.push(outboundLeg); // Add the outbound leg
      allFlightResults.push(flightResults); // Add the result

      // Get return itinerary
      const returnItinerary = offer.itineraries[1];

      // Outbound Itinerary
      const returnLeg: FlightLeg = {
        direction: "return",
        legDuration: returnItinerary.duration,
        segments: [],
      };
      // Loop through the segments of the leg
      for (const flight of returnItinerary.segments) {
        const duration = flight.duration;
        const departure = flight.departure;
        const arrival = flight.arrival;
        const carrierCode = flight.carrierCode;

        const airlineName = carrierMap[carrierCode] ?? carrierCode;

        const flightSegment: FlightSegment = {
          duration: duration,
          departure: {
            airport: departure.iataCode,
            time: departure.at,
          },
          arrival: {
            airport: arrival.iataCode,
            time: arrival.at,
          },
          airline: airlineName,
        };

        returnLeg.segments.push(flightSegment);
      }
      flightResults.legs.push(returnLeg); // Add the outbound leg
      allFlightResults.push(flightResults); // Add the result
    }

    // console.log(JSON.stringify(allFlightResults, null, 2));
    return JSON.stringify(allFlightResults); // Tool must always return a string
  },
  {
    name: "searchFlights",
    description: `
    Searches for the cheapest available round-trip flights between two airports.

    Use this tool when the user asks about:
    - Flight prices or availability
    - Round-trip airfare
    - Flight options between destinations

    REQUIRED PARAMETERS (all must be provided):
    - Origin airport (IATA code, e.g., "JFK", "LAX", "ORD")
    - Destination airport (IATA code, e.g., "LHR", "CDG", "NRT")
    - Departure date (format: YYYY-MM-DD)
    - Return date (format: YYYY-MM-DD)

    If any required parameter is missing from the conversation:
    - Ask the user for it naturally
    - Explain what format you need (e.g., "I need the 3-letter airport code")
    `,
    schema: z.object({
      originLocationCode: z
        .string()
        .length(3)
        .describe(
          "REQUIRED: IATA airport code for departure (exactly 3 letters, e.g., JFK, LAX, ORD)",
        ),
      destinationLocationCode: z
        .string()
        .length(3)
        .describe(
          "REQUIRED: IATA airport code for destination (exactly 3 letters, e.g., LHR, CDG, NRT)",
        ),
      departureDate: z
        .string()
        .describe(
          "REQUIRED: Departure date in YYYY-MM-DD format (e.g., 2026-01-08)",
        ),
      returnDate: z
        .string()
        .describe(
          "REQUIRED: Return date in YYYY-MM-DD format (e.g., 2026-01-16)",
        ),
      includedAirlinesCodes: z
        .array(z.string().length(2))
        .optional()
        .describe(
          "OPTIONAL: Array of 2-letter airline codes (e.g., ['AA', 'DL', 'UA'])",
        ),
      cityName: z
        .string()
        .describe(
          'REQUIRED: Name of the destination city (e.g. "Barcelona"). Used to fetch city-centre coordinates.',
        ),
    }),
  },
);
