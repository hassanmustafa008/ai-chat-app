import "dotenv/config";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { HumanMessage } from "@langchain/core/messages";

const getWeather = tool(
  async ({ city }) => {
    // Fake data for now — we're testing the WIRING, not building a real weather API yet
    const fakeData: Record<string, string> = {
      karachi: "34°C, sunny",
      london: "15°C, rainy",
    };
    return fakeData[city.toLowerCase()] ?? "No data for that city";
  },
  {
    name: "get_weather",
    description: "Get the current weather for a given city",
    schema: z.object({
      city: z.string().describe("The city name, e.g. 'London'"),
    }),
  },
);

const model = new ChatGoogleGenerativeAI({
  model: "gemini-3.1-flash-lite",
  apiKey: process.env.GOOGLE_API_KEY || "",
  temperature: 0,
});

const modelWithTools = model.bindTools([getWeather]);

async function main() {
  const response = await modelWithTools.invoke([
    new HumanMessage("Is it raining in London right now?"),
  ]);

  console.log("content:", response.content);
  console.log("tool_calls:", JSON.stringify(response.tool_calls, null, 2));
}

main();
