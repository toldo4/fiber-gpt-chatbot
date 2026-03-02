'use client'

import { useEffect, useState } from 'react'

interface LogEntry {
  timestamp: string
  question: string
  answer: string | object
  sources: string[]
  modelId: string
}

function safeAnswer(answer: string | object): string {
  if (typeof answer === 'string') return answer
  if (answer && typeof answer === 'object') {
    const obj = answer as Record<string, unknown>
    if (typeof obj.text === 'string') return obj.text
    return JSON.stringify(obj, null, 2)
  }
  return String(answer ?? '')
}

export function LogsViewer() {
  const today = new Date().toISOString().slice(0, 10)
  const [date, setDate] = useState(today)
  const [entries, setEntries] = useState<LogEntry[]>([])
  const [loading, setLoading] = useState(false)
  const [expanded, setExpanded] = useState<number | null>(null)

  useEffect(() => {
    setLoading(true)
    fetch(`/api/logs?date=${date}`)
      .then(r => r.json())
      .then(data => setEntries(data.entries ?? []))
      .finally(() => setLoading(false))
  }, [date])

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Chat Logs</h1>
        <input
          type="date"
          value={date}
          max={today}
          onChange={e => setDate(e.target.value)}
          className="border rounded-md px-3 py-1.5 text-sm bg-background"
        />
      </div>

      {/* Stats bar */}
      <div className="text-sm text-muted-foreground">
        {loading ? 'Loading...' : `${entries.length} conversation${entries.length !== 1 ? 's' : ''} on ${date}`}
      </div>

      {/* Entries */}
      {!loading && entries.length === 0 && (
        <div className="text-center py-16 text-muted-foreground">
          No logs found for this date.
        </div>
      )}

      <div className="space-y-3">
        {entries.map((entry, i) => (
          <div key={i} className="border rounded-lg overflow-hidden">
            {/* Question row — always visible */}
            <button
              className="w-full text-left px-4 py-3 flex items-start gap-3 hover:bg-muted/50 transition-colors"
              onClick={() => setExpanded(expanded === i ? null : i)}
            >
              <span className="text-xs text-muted-foreground mt-0.5 shrink-0">
                {new Date(entry.timestamp).toLocaleTimeString()}
              </span>
              <span className="font-medium text-sm flex-1 line-clamp-2">
                {entry.question}
              </span>
              <span className="text-muted-foreground text-xs shrink-0">
                {expanded === i ? '▲' : '▼'}
              </span>
            </button>

            {/* Answer — expanded */}
            {expanded === i && (
              <div className="px-4 pb-4 space-y-3 border-t bg-muted/20">
                <div className="pt-3">
                  <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1">Answer</p>
                  <p className="text-sm whitespace-pre-wrap">{safeAnswer(entry.answer)}</p>
                </div>

                {entry.sources?.length > 0 && (
                  <div>
                    <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1">Sources</p>
                    <ul className="space-y-0.5">
                      {entry.sources.map((s, j) => (
                        <li key={j} className="text-xs text-muted-foreground">• {s}</li>
                      ))}
                    </ul>
                  </div>
                )}

                <p className="text-xs text-muted-foreground">Model: {entry.modelId}</p>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}