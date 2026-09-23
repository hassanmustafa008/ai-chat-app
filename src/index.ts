import "dotenv/config";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import {
  SystemMessage,
  HumanMessage,
  AIMessage,
  BaseMessage
} from "@langchain/core/messages";
import * as readline from "node:readline/promises";

const apiKey = process.env.GOOGLE_API_KEY;
if (!apiKey) {
  throw new Error("GOOGLE_API_KEY is not set in .env");
}

const model = new ChatGoogleGenerativeAI({
  model: "gemini-3.6-flash",
  apiKey: apiKey,
  temperature: 0.7,
});

const history: BaseMessage[] = [
  new SystemMessage("You are a concise, friendly assistant."),
];

async function main() {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

  console.log("Chat started. Type 'exit' to quit.\n");

  while (true) {
    const userInput = await rl.question("You: ");
    if (userInput.trim().toLowerCase() === "exit") break;

    history.push(new HumanMessage(userInput));

    const response = await model.invoke(history);
    console.log("AI:", response.content, "\n");

    history.push(new AIMessage(response.content as string));
  }

  rl.close();
}

main();
