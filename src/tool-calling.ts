import "dotenv/config";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import {
  HumanMessage,
  ToolMessage,
  BaseMessage,
} from "@langchain/core/messages";

const getWeather = tool(
  async ({ city }) => {
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
  const messages: BaseMessage[] = [
    new HumanMessage("Is it raining in London right now?"),
  ];

  const response = await modelWithTools.invoke(messages);
  messages.push(response); // the model's tool-call request itself becomes part of history

  if (response.tool_calls && response.tool_calls.length > 0) {
    for (const call of response.tool_calls) {
      console.log(`Executing tool: ${call.name}(${JSON.stringify(call.args)})`);

      const result = await getWeather.invoke(call.args as { city: string });

      messages.push(
        new ToolMessage({
          content: result,
          tool_call_id: call.id!,
        }),
      );
    }

    // Send the tool's result back so the model can produce a real answer
    const finalResponse = await modelWithTools.invoke(messages);
    console.log("Final answer:", finalResponse.content);
  } else {
    console.log("Direct answer:", response.content);
  }
}

main();
