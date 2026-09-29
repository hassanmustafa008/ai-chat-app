import "dotenv/config";
import { GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GOOGLE_API_KEY||"",
  model: "gemini-embedding-2",
});

const model = new ChatGoogleGenerativeAI({
  model: "gemini-3.1-flash-lite",
  apiKey: process.env.GOOGLE_API_KEY||"",
  temperature: 0,
});

const docs = [
  new Document({ pageContent: "The Eiffel Tower is located in Paris, France." }),
  new Document({ pageContent: "Sushi is a traditional Japanese dish made with rice and fish." }),
  new Document({ pageContent: "The Great Wall of China stretches over 13,000 miles." }),
  new Document({ pageContent: "Python is a popular programming language for data science." }),
];

async function ragQuery(question: string) {
  const store = await MemoryVectorStore.fromDocuments(docs, embeddings);
  const retrieved = await store.similaritySearch(question, 2);

  const context = retrieved.map((d) => d.pageContent).join("\n");

  const response = await model.invoke([
    new SystemMessage(
      `Answer the question using ONLY the context below. If the context doesn't contain the answer, say so.\n\nContext:\n${context}`
    ),
    new HumanMessage(question),
  ]);

  return response.content;
}

async function main() {
  console.log(await ragQuery("Where is the Eiffel Tower?"));
  console.log("---");
  console.log(await ragQuery("What's the best programming language for web design?"));
}

main();