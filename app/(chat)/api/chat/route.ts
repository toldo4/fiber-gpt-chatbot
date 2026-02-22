import {
  type Message,
  convertToCoreMessages,
  createDataStreamResponse,
  smoothStream,
  streamText,
} from 'ai'

import { customModel } from '@/lib/ai'
import { models } from '@/lib/ai/models'
import { generateUUID, getMostRecentUserMessage } from '@/lib/utils'
import { 
  searchRAG, 
  buildContext, 
  extractSourceFiles,
  isCorpusMetaQuestion 
} from '@/lib/rag/search'
import { getCorpusStats } from '@/lib/rag/index-loader'
import { getPaperMetadata, formatPaperCitation } from '@/lib/rag/paper-metadata'

export const maxDuration = 60

// System prompt matching your Express server exactly
const RAG_SYSTEM_PROMPT = 
  'You are a cranberry chemistry/analytics assistant. ' +
  'Use ONLY the provided CONTEXT to answer. If unsure, say so. Keep answers concise. ' +
  'Provide all source citations at the end of your response in a "Sources:" section.'

export async function POST(request: Request) {
  const {
    id,
    messages,
    modelId,
  }: { id: string; messages: Array<Message>; modelId: string } =
    await request.json()

  const model = models.find((model) => model.id === modelId)

  if (!model) {
    return new Response('Model not found', { status: 404 })
  }

  const coreMessages = convertToCoreMessages(messages)
  const userMessage = getMostRecentUserMessage(coreMessages)

  if (!userMessage) {
    return new Response('No user message found', { status: 400 })
  }

  const userMessageId = generateUUID()

  // Extract query from user message
  const query = typeof userMessage.content === 'string' 
    ? userMessage.content 
    : userMessage.content.map(c => c.type === 'text' ? c.text : '').join(' ')

  return createDataStreamResponse({
    execute: async (dataStream) => {
      dataStream.writeData({
        type: 'user-message-id',
        content: userMessageId,
      })

      try {
        // Check for corpus meta-questions (like "how many documents")
        if (isCorpusMetaQuestion(query)) {
          const stats = getCorpusStats()
          const paperList = stats.papers
            .map(paper => `- ${paper.title}\n  Journal: ${paper.journal} (${paper.year})\n  DOI: ${paper.doi}`)
            .join('\n\n')
          
          const response = `There are ${stats.total_documents} research papers indexed.\n\n${paperList}\n`
          
          dataStream.writeData({
            type: 'text',
            content: response
          })
          
          return
        }

        // Perform RAG search
        const searchResults = await searchRAG(query, 10)
        const context = buildContext(searchResults)
        const sourceFilenames = extractSourceFiles(searchResults)

        // Format sources with full citations
        const formattedSources = sourceFilenames
          .map(filename => {
            const metadata = getPaperMetadata(filename)
            if (metadata) {
              return formatPaperCitation(metadata)
            }
            return filename
          })
          .join('\n\n')

        // Build the prompt with formatted citations
        const ragPrompt = 
          `CONTEXT:\n${context}\n\nQUESTION:\n${query}\n\n` +
          `INSTRUCTIONS:\n- Base your answer only on CONTEXT.\n- Answer the question naturally without inline citations.\n- After your answer, add a "Sources:" section listing all papers you referenced.\n- Format each source as: Title. Journal (Year). DOI: [doi]\n\nAVAILABLE SOURCES:\n${formattedSources}\n`

        // Stream the response
        const result = streamText({
          model: customModel(model.apiIdentifier),
          system: RAG_SYSTEM_PROMPT,
          messages: [
            { role: 'user', content: ragPrompt }
          ],
          experimental_transform: smoothStream({ chunking: 'word' }),
          experimental_telemetry: {
            isEnabled: true,
            functionId: 'stream-text-rag',
          },
          onFinish: async () => {
            // Append sources with metadata after streaming completes
            const sourcesWithMetadata = sourceFilenames.map(filename => {
              const metadata = getPaperMetadata(filename)
              return metadata || { filename }
            })
            
            dataStream.writeData({
              type: 'sources',
              content: JSON.stringify({ 
                files: sourceFilenames,
                metadata: sourcesWithMetadata 
              })
            })
          }
        })

        // Merge the stream into data stream
        result.mergeIntoDataStream(dataStream)
      } catch (error) {
        console.error('RAG error:', error)
        
        dataStream.writeData({
          type: 'error',
          content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`
        })
      }
    },
  })
}