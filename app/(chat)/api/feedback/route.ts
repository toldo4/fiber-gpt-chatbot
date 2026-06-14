import { logFeedback } from '@/lib/logger'

export async function POST(request: Request) {
  const { messageId, chatId, rating, messagePreview } = await request.json()

  if (!messageId || !chatId || !['up', 'down'].includes(rating)) {
    return new Response('Invalid payload', { status: 400 })
  }

  logFeedback({ messageId, chatId, rating, messagePreview: messagePreview ?? '' })

  return new Response(null, { status: 204 })
}
