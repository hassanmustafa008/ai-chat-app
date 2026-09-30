import "dotenv/config";
import { readFileSync } from "node:fs";
import { GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GOOGLE_API_KEY|| "",
  model: "gemini-embedding-2",
});

const model = new ChatGoogleGenerativeAI({
  model: "gemini-3.1-flash-lite",
  apiKey: process.env.GOOGLE_API_KEY||"",
  temperature: 0,
});

async function buildStore() {
  const rawText = readFileSync("src/sample-doc.txt", "utf-8");

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 300,
    chunkOverlap: 50,
  });

  const chunks = await splitter.splitText(rawText);
  console.log(`Split into ${chunks.length} chunks:\n`);
  chunks.forEach((c, i) => console.log(`--- chunk ${i} ---\n${c}\n`));

  const docs = chunks.map((c) => new Document({ pageContent: c }));
  return MemoryVectorStore.fromDocuments(docs, embeddings);
}

async function ragQuery(store: MemoryVectorStore, question: string) {
  const retrieved = await store.similaritySearch(question, 2);
  const context = retrieved.map((d) => d.pageContent).join("\n\n");

  const response = await model.invoke([
    new SystemMessage(
      `Answer using ONLY the context below. If it doesn't contain the answer, say so.\n\nContext:\n${context}`
    ),
    new HumanMessage(question),
  ]);

  return response.content;
}

async function main() {
  const store = await buildStore();

  console.log("Q: How long do I have to return something?");
  console.log(await ragQuery(store, "How long do I have to return something?"));
  console.log("\nQ: Can I call support on weekends?");
  console.log(await ragQuery(store, "Can I call support on weekends?"));
}

main();