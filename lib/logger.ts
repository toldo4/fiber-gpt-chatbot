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
}) {
  const line = JSON.stringify({
    timestamp: new Date().toISOString(),
    ...entry,
  })

  const logFile = getLogFile()
  fs.mkdirSync(path.dirname(logFile), { recursive: true })
  fs.appendFileSync(logFile, line + '\n', 'utf8')
}