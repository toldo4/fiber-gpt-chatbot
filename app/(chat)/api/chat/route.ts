import {
  type Message,
  convertToCoreMessages,
  createDataStreamResponse,
  streamText,
} from 'ai'

import { customModel } from '@/lib/ai'
import { models } from '@/lib/ai/models'
import { generateUUID, getMostRecentUserMessage } from '@/lib/utils'
import {
  searchRAG,
  extractSourceFiles,
  isCorpusMetaQuestion
} from '@/lib/rag/search'
import { getCorpusStats } from '@/lib/rag/index-loader'
import { getPaperMetadata, formatPaperCitation } from '@/lib/rag/paper-metadata'
import { logQA } from '@/lib/logger'

export const maxDuration = 60

// Generic RAG assistant system prompt with inline citation support
const RAG_SYSTEM_PROMPT =
  'You are a research assistant with access to a corpus of academic papers. ' +
  'Use ONLY the provided CONTEXT to answer questions. If the context does not contain enough information, say so clearly. ' +
  'Prioritize information from Methods, Results, and Discussion sections over Introduction sections, which typically contain general background rather than specific findings. ' +
  'When answering, cite your sources inline using numbered brackets like [1], [2], etc., corresponding to the numbered sources listed at the end. ' +
  'You may cite the same source multiple times. ' +
  'End every response with a "Sources:" section that lists each cited paper in order, formatted as: [N] Title. Journal (Year). DOI/Link: identifier'

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
            .map(paper => {
              const identifier = paper.doi ? `DOI: ${paper.doi}` : paper.link ? `Link: ${paper.link}` : 'No identifier';
              return `- ${paper.title}\n  Journal: ${paper.journal} (${paper.year})\n  ${identifier}`;
            })
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
        const sourceFilenames = extractSourceFiles(searchResults)

        // Build context using file-based citation numbers (aligned with AVAILABLE SOURCES)
        const context = searchResults.map(r => {
          const fileIndex = sourceFilenames.indexOf(r.file) + 1
          const rawSection: string = (r as any).section ?? 'General'
          const section = rawSection.length > 40 ? rawSection.slice(0, 37).trimEnd() + '...' : rawSection
          return `[Source ${fileIndex}][${section}]\n${r.text}`
        }).join('\n\n')

        // Build numbered source list for inline citation
        const numberedSources = sourceFilenames.map((filename, index) => {
          const metadata = getPaperMetadata(filename)
          const citation = metadata ? formatPaperCitation(metadata) : filename
          return `[${index + 1}] ${citation}`
        }).join('\n\n')

        // Build the prompt instructing the LLM to use numbered inline citations
        const sourceCount = sourceFilenames.length
        const ragPrompt =
          `CONTEXT:\n${context}\n\n` +
          `QUESTION:\n${query}\n\n` +
          `AVAILABLE SOURCES (${sourceCount} total, numbered [1] to [${sourceCount}]):\n${numberedSources}\n\n` +
          `INSTRUCTIONS:\n` +
          `- Answer using ONLY the information in CONTEXT above.\n` +
          `- Give less weight to content from Introduction sections, which tend to be general background. Prefer specific findings from Methods, Results, and Discussion sections.\n` +
          `- Try to be as concise as possible while still answering the question fully. Avoid unnecessary elaboration or repetition.\n` +
          `- Cite sources inline using the format [N][Section], where N is the source number and Section is the section name from the context header (e.g. [Methods]). Example: [2][Methods], [1][Results].\n` +
          `- CRITICAL: You MUST only use citation numbers that exist in the AVAILABLE SOURCES list above (between [1] and [${sourceCount}]). Do NOT invent or use any citation number outside this range.\n` +
          `- If only one source exists, only ever cite [1]. If two exist, only cite [1] or [2]. Never cite a number higher than ${sourceCount}.\n` +
          `- Multiple citations can be used together only if each number exists, e.g. [1][Methods][2][Results].\n` +
          `- At the end of your response, include a "Sources:" section listing only the papers you actually cited.\n`

        // Stream the response
        const result = streamText({
          model: customModel(model.apiIdentifier),
          ...(model.temperature !== undefined ? { temperature: model.temperature } : {}),
          system: RAG_SYSTEM_PROMPT,
          messages: [
            { role: 'user', content: ragPrompt }
          ],
          experimental_telemetry: {
            isEnabled: true,
            functionId: 'stream-text-rag',
          },
          onFinish: async (result) => {
            const text = result.text

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

            logQA({
              question: query,
              answer: text,
              sources: sourceFilenames,
              modelId: model.apiIdentifier,
              retrievedChunks: searchResults.map(r => ({
                file: r.file,
                title: getPaperMetadata(r.file)?.title,
                section: r.section,
                preview: r.text.slice(0, 120).replace(/\s+/g, ' ').trim(),
              })),
            })
          }
        })

        result.mergeIntoDataStream(dataStream)
        await result.text
      } catch (error) {
        console.error('RAG error:', error)

        dataStream.writeData({
          type: 'error',
          content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`
        })
      }
    },
    onError: (error) => {
      console.error('Stream error:', error);
      return error instanceof Error ? error.message : String(error);
    }
  })
}