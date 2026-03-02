import fs from 'fs'
import path from 'path'
import { NextResponse } from 'next/server'

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const date = searchParams.get('date') ?? new Date().toISOString().slice(0, 10)

  const logFile = path.join(process.cwd(), 'logs', `chat-log-${date}.jsonl`)

  if (!fs.existsSync(logFile)) {
    return NextResponse.json({ date, entries: [] })
  }

  const lines = fs.readFileSync(logFile, 'utf8').trim().split('\n').filter(Boolean)
  const entries = lines.map(line => JSON.parse(line)).reverse() // newest first

  return NextResponse.json({ date, entries })
}