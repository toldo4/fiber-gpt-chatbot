// app/api/rag-stats/route.ts
import { NextResponse } from 'next/server'
import { getCorpusStats } from '@/lib/rag/index-loader'

export const runtime = 'nodejs'

export async function GET() {
  try {
    const stats = getCorpusStats()
    
    return NextResponse.json({
      success: true,
      ...stats
    })
  } catch (error) {
    console.error('Error getting corpus stats:', error)
    
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    )
  }
}