import "dotenv/config";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Document } from "@langchain/core/documents";

const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GOOGLE_API_KEY || "",
  model: "gemini-embedding-2",
});

const docs = [
  new Document({ pageContent: "The Eiffel Tower is located in Paris, France." }),
  new Document({ pageContent: "Sushi is a traditional Japanese dish made with rice and fish." }),
  new Document({ pageContent: "The Great Wall of China stretches over 13,000 miles." }),
  new Document({ pageContent: "Python is a popular programming language for data science." }),
];

async function main() {
  const store = await MemoryVectorStore.fromDocuments(docs, embeddings);

  console.log("embeddings generated: ", store.memoryVectors[0]?.embedding);

  const results = await store.similaritySearchWithScore("famous landmarks in Europe", 2);

  for (const [doc, score] of results) {
    console.log(score.toFixed(4), "-", doc.pageContent);
  }
}

main();