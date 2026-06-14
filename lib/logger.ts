import fs from 'fs'
import path from 'path'

function getLogFile(): string {
  const date = new Date().toISOString().slice(0, 10) // "2026-03-02"
  return path.join(process.cwd(), 'logs', `chat-log-${date}.jsonl`)
}

export function logQA(entry: {
  question: string
  answer: string
  sources: string[]
  modelId: string
  retrievedChunks?: { file: string; title?: string; section?: string; preview: string }[]
}) {
  const line = JSON.stringify({
    timestamp: new Date().toISOString(),
    type: 'qa',
    ...entry,
  })

  const logFile = getLogFile()
  fs.mkdirSync(path.dirname(logFile), { recursive: true })
  fs.appendFileSync(logFile, line + '\n', 'utf8')
}

export function logFeedback(entry: {
  messageId: string
  chatId: string
  rating: 'up' | 'down'
  messagePreview: string
}) {
  const line = JSON.stringify({
    timestamp: new Date().toISOString(),
    type: 'feedback',
    ...entry,
  })

  const logFile = getLogFile()
  fs.mkdirSync(path.dirname(logFile), { recursive: true })
  fs.appendFileSync(logFile, line + '\n', 'utf8')
}