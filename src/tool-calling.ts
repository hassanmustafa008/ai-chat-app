import "dotenv/config";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import {
  HumanMessage,
  ToolMessage,
  BaseMessage,
  AIMessage,
} from "@langchain/core/messages";

const searchWeb = tool(
  async ({ query }) => {
    const res = await fetch("https://api.tavily.com/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: process.env.TAVILY_API_KEY, query, max_results: 3 }),
    });
    const data = await res.json();
    console.log("Tavily search results:", data);
    return data.results.map((r: any) => `${r.title}: ${r.content}`).join("\n\n");
  },
  {
    name: "search_web",
    description:
      "Search the web for current information not available in training data — recent news, current events, or anything after the model's knowledge cutoff",
    schema: z.object({ query: z.string().describe("The search query") }),
  }
);

const getWeather = tool(
  async ({ city }) => {
    try {
      const geoRes = await fetch(
        `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1`
      );
      const geoData = await geoRes.json();
      const location = geoData.results?.[0];

      if (!location) {
        return `Could not find a location named "${city}"`;
      }

      const { latitude, longitude, name } = location;
      const weatherRes = await fetch(
        `https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current=temperature_2m`
      );
      const weatherData = await weatherRes.json();
      const temp = weatherData.current?.temperature_2m;

      return `${temp}°C in ${name}`;
    } catch (err) {
      return `Error fetching weather for "${city}": ${(err as Error).message}`;
    }
  },
  {
    name: "get_weather",
    description: "Get the current real-time temperature for a given city in Celsius",
    schema: z.object({ city: z.string().describe("The city name") }),
  }
);

const multiply = tool(async ({ a, b }) => String(a * b), {
  name: "multiply",
  description: "Multiply two numbers",
  schema: z.object({ a: z.number(), b: z.number() }),
});

const tools = [getWeather, multiply, searchWeb];
const toolsByName = Object.fromEntries(tools.map((t) => [t.name, t]));

const model = new ChatGoogleGenerativeAI({
  model: "gemini-3.1-flash-lite",
  apiKey: process.env.GOOGLE_API_KEY || " ",
  temperature: 0,
});

const modelWithTools = model.bindTools(tools);

async function runAgent(question: string) {
  const messages: BaseMessage[] = [new HumanMessage(question)];

  const MAX_ITERATIONS = 5;
  for (let i = 0; i < MAX_ITERATIONS; i++) {
    const response = await modelWithTools.invoke(messages);
    messages.push(response);

    if (!response.tool_calls || response.tool_calls.length === 0) {
      return response.content; // model is done — no more tools needed
    }

    for (const call of response.tool_calls) {
      console.log(
        `  [round ${i + 1}] calling ${call.name}(${JSON.stringify(call.args)})`,
      );
      const selectedTool = toolsByName[call.name];
      const result = await selectedTool.invoke(call.args as any);

      messages.push(
        new ToolMessage({ content: result, tool_call_id: call.id! }),
      );
    }
  }

  throw new Error("Agent did not finish within MAX_ITERATIONS");
}

async function main() {
  const answer = await runAgent(
    "search the web for who won the most recent Nobel Prize in Physics.",
  );
  console.log("Final answer:", answer);
}

main();
