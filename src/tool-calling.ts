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

const getWeather = tool(
  async ({ city }) => {
    const fakeData: Record<string, number> = { karachi: 34, london: 15 };
    const temp = fakeData[city.toLowerCase()];
    return temp !== undefined ? `${temp}°C` : "No data for that city";
  },
  {
    name: "get_weather",
    description: "Get the current temperature for a given city in Celsius",
    schema: z.object({ city: z.string().describe("The city name") }),
  },
);

const multiply = tool(async ({ a, b }) => String(a * b), {
  name: "multiply",
  description: "Multiply two numbers",
  schema: z.object({ a: z.number(), b: z.number() }),
});

const tools = [getWeather, multiply];
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
    "What's the temperature in Karachi multiplied by the temperature in London?",
  );
  console.log("Final answer:", answer);
}

main();
