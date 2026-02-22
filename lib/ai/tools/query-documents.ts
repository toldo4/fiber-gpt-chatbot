// lib/ai/tools/query-documents.ts
import { tool } from 'ai'
import { z } from 'zod'
import { searchRAG, buildContext, extractSourceFiles } from '@/lib/rag/search'

export const queryDocuments = tool({
  description: 'Search through cranberry research documents to find relevant information. Use this when the user asks questions about cranberries, chemistry, analytics, or references the document corpus.',
  parameters: z.object({
    query: z.string().describe('The search query to find relevant document chunks'),
    topK: z.number().optional().default(10).describe('Number of relevant chunks to retrieve (default: 10)')
  }),
  execute: async ({ query, topK }) => {
    try {
      // Perform RAG search
      const results = await searchRAG(query, topK)
      
      if (results.length === 0) {
        return {
          success: false,
          message: 'No relevant documents found for this query.',
          context: '',
          sources: []
        }
      }
      
      // Build context and extract sources
      const context = buildContext(results)
      const sources = extractSourceFiles(results)
      
      return {
        success: true,
        message: `Found ${results.length} relevant document chunks from ${sources.length} source(s).`,
        context,
        sources,
        topScore: results[0]?.score || 0
      }
    } catch (error) {
      console.error('Error querying documents:', error)
      return {
        success: false,
        message: `Error searching documents: ${error instanceof Error ? error.message : 'Unknown error'}`,
        context: '',
        sources: []
      }
    }
  }
})