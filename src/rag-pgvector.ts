import "dotenv/config";
import { readFileSync } from "node:fs";
import { GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { PGVectorStore } from "@langchain/pgvector";
import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GOOGLE_API_KEY || "",
  model: "gemini-embedding-2",
});

const model = new ChatGoogleGenerativeAI({
  model: "gemini-3.1-flash-lite",
  apiKey: process.env.GOOGLE_API_KEY || "",
  temperature: 0,
});

async function getStore() {
  return PGVectorStore.initialize(embeddings, {
    postgresConnectionOptions: {
      host: "localhost",
      port: 5433,
      user: "raguser",
      password: "ragpass",
      database: "ragdb",
    },
    tableName: "doc_chunks",
  });
}

async function ingest(store: PGVectorStore) {
  const rawText = readFileSync("src/sample-doc.txt", "utf-8");
  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 300, chunkOverlap: 50 });
  const chunks = await splitter.splitText(rawText);
  const docs = chunks.map((c) => new Document({ pageContent: c }));
  await store.addDocuments(docs);
  console.log(`Ingested ${docs.length} chunks into Postgres.`);
}

async function ragQuery(store: PGVectorStore, question: string) {
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
  const store = await getStore();

  if (process.argv[2] === "ingest") {
    await ingest(store);
  } else {
    console.log(await ragQuery(store, "What's my favorite color?"));
  }

  await store.end();
}

main();