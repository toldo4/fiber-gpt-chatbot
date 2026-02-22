// lib/rag/search.ts
import { openai } from '@ai-sdk/openai'
import { embed } from 'ai'
import { loadSplitIndex, type RagChunk } from './index-loader'

export interface SearchResult extends RagChunk {
  score: number
}

function cosine(a: number[], b: number[]): number {
  let dot = 0
  let na = 0
  let nb = 0
  
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12)
}

export async function searchRAG(
  query: string,
  topK: number = 10
): Promise<SearchResult[]> {
  const index = loadSplitIndex()
  
  // Embed the query using the same model as the index
  const { embedding } = await embed({
    model: openai.embedding('text-embedding-3-small'),
    value: query,
  })
  
  // Rank chunks by cosine similarity
  const scored = index.chunks.map(chunk => ({
    ...chunk,
    score: cosine(embedding, chunk.embedding)
  }))
  
  scored.sort((a, b) => b.score - a.score)
  
  return scored.slice(0, topK)
}

export function buildContext(results: SearchResult[]): string {
  return results
    .map((r, i) => `[[Source ${i + 1}: ${r.file}]]\n${r.text}`)
    .join('\n\n')
}

export function extractSourceFiles(results: SearchResult[]): string[] {
  return [...new Set(results.map(r => r.file))]
}

export function isCorpusMetaQuestion(query: string): boolean {
  const corpusQ = /(how many|number of|count of).*(documents|docs|files|sources)/i
  return corpusQ.test(query)
}