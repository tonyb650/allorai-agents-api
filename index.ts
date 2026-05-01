import "dotenv/config";
import express from "express";
import { graph } from "./graph/index.js";
import type { BaseMessage } from "@langchain/core/messages";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import type { ChatRequest, ChatResponse, Message, TipsRequest, TipsResponse } from "./types/api.js";
import { createEmptyTrip } from "./types/trip.js";
import { generateTips } from "./graph/nodes/tips/tipsNode.js";

type Request = express.Request;
type Response = express.Response;
type NextFunction = express.NextFunction;

const app = express();
const PORT = process.env.PORT || 8000;

app.use(express.json());

// Request logging middleware
app.use((req: Request, res: Response, next: NextFunction) => {
  const start = Date.now();
  console.log(`[${new Date().toISOString()}] --> ${req.method} ${req.path}`);

  if (req.body && Object.keys(req.body).length > 0) {
    console.log("Request body:", req.body);
  }

  res.on("finish", () => {
    const duration = Date.now() - start;
    console.log(
      `[${new Date().toISOString()}] <-- ${req.method} ${req.path} ${res.statusCode} (${duration}ms)`,
    );
  });

  next();
});

  app.use((req, res, next) => {
    if (req.path === "/health") return next();                                                                                                                          
    if (req.headers["x-api-key"] !== process.env.INTERNAL_API_KEY) {                                                                                                    
      return res.status(401).json({ error: "unauthorized" });       
    }                                                                                                                                                                   
    next();                                                                                                                                                           
  }); 

app.get("/health", (req: Request, res: Response) => {
  res.json({ status: "ok" });
});

function toMessages(messages: BaseMessage[]): Message[] {
  return messages
    .map((m) => ({
      type: m.getType() as "human" | "ai",
      content:
        typeof m.content === "string" ? m.content : JSON.stringify(m.content),
    }))
    .filter((m): m is Message => m.type === "human" || m.type === "ai");
}

function filterForClient(messages: Message[]): Message[] {
  return messages.filter((m) => {
    if (m.type === "ai" && (!m.content || m.content.trim() === ""))
      return false;
    return true;
  });
}

const MAX_MESSAGES = 30;
const MAX_MESSAGE_CONTENT_LENGTH = 4000;
const GRAPH_RECURSION_LIMIT = 10;

app.post("/chat", async (req: Request, res: Response) => {
  try {
    const { messages: inputMessages, trip, data } = req.body as ChatRequest;

    if (!inputMessages || !Array.isArray(inputMessages)) {
      res.status(400).json({
        error: "Invalid request. Expected { messages: Array, trip: Trip }",
      });
      return;
    }

    if (inputMessages.length > MAX_MESSAGES) {
      res.status(413).json({ error: `Too many messages (max ${MAX_MESSAGES}).` });
      return;
    }

    if (inputMessages.some((m) => typeof m.content !== "string" || m.content.length > MAX_MESSAGE_CONTENT_LENGTH)) {
      res.status(413).json({ error: `Message content too long (max ${MAX_MESSAGE_CONTENT_LENGTH} chars).` });
      return;
    }

    // Convert frontend messages to LangChain messages
    const conversation: BaseMessage[] = inputMessages.map((m) => {
      if (m.type === "human") {
        return new HumanMessage(m.content);
      }
      return new AIMessage(m.content);
    });

    // Invoke graph with trip context
    const result = await graph.invoke(
      {
        messages: conversation,
        trip: trip || createEmptyTrip(),
        data: data || null,
      },
      { recursionLimit: GRAPH_RECURSION_LIMIT },
    );

    // Convert all messages to response format (filters out tool messages)
    const allMessages = toMessages(result.messages);

    // Filter messages for client (remove tool messages and empty AI messages)
    const filteredMessages = filterForClient(allMessages);

    const response: ChatResponse = {
      messages: filteredMessages,
      data: result.data || null,
      trip: result.trip || trip || createEmptyTrip(),
    };

    res.json(response);
  } catch (error) {
    console.error("Error processing chat:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.post("/tips", async (req: Request, res: Response) => {
  try {
    const { trip } = req.body as TipsRequest;

    if (!trip?.destination) {
      res.status(400).json({
        error: "Missing trip destination. A destination is required to generate tips.",
      });
      return;
    }

    const tips = await generateTips(trip);

    const response: TipsResponse = {
      data: {
        type: "tips",
        options: tips,
      },
      trip: trip || createEmptyTrip(),
    };

    res.json(response);
  } catch (error) {
    console.error("Error generating tips:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.listen(PORT, () => {
  console.log(`\nServer running on http://localhost:${PORT}`);
  console.log(`Press Ctrl+C to stop\n`);
});
