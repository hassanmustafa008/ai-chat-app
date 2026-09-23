import "dotenv/config";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import {
  SystemMessage,
  HumanMessage,
  AIMessage,
  BaseMessage
} from "@langchain/core/messages";

const apiKey = process.env.GOOGLE_API_KEY;
if (!apiKey) {
  throw new Error("GOOGLE_API_KEY is not set in .env");
}

const model = new ChatGoogleGenerativeAI({
  model: "gemini-3.6-flash",
  apiKey: apiKey,
  temperature: 0.7,
});

async function main() {
  const messages: BaseMessage[] = [
    new SystemMessage("You are a concise assistant. Answer in one sentence."),
    new HumanMessage("What's the capital of France?"),
  ];

  const response1 = await model.invoke(messages);
  console.log("AI:", response1.content);

  // Append the AI's reply and ask a follow-up that depends on prior context
  messages.push(new AIMessage(response1.content as string));
  messages.push(new HumanMessage("What's a famous landmark there?"));

  const response2 = await model.invoke(messages);
  console.log("AI:", response2.content);
}

main();
