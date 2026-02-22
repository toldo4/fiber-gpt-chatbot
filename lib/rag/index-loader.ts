// lib/rag/index-loader.ts
import fs from 'fs'
import path from 'path'
import { getPaperMetadata, type PaperMetadata } from './paper-metadata'

export interface RagChunk {
  id: string
  file: string
  text: string
  embedding: number[]
}

export interface RagIndex {
  created_at: string
  model: string
  chunks: RagChunk[]
}

// Cache the loaded index in memory
let indexCache: RagIndex | null = null
let indexLoadTime: number | null = null
const INDEX_CACHE_TTL = 60 * 60 * 1000 // 1 hour

export function loadSplitIndex(forceReload = false): RagIndex {
  // Return cached index if still valid
  if (!forceReload && indexCache && indexLoadTime) {
    if (Date.now() - indexLoadTime < INDEX_CACHE_TTL) {
      return indexCache
    }
  }

  const partsDir = path.join(process.cwd(), 'index-parts')
  
  if (!fs.existsSync(partsDir)) {
    throw new Error('No index-parts/ directory found. Run: node build-index.js')
  }

  const partFiles = fs.readdirSync(partsDir)
    .filter(f => f.startsWith('index-part-') && f.endsWith('.json'))
    .sort()

  if (partFiles.length === 0) {
    throw new Error('No index parts found in index-parts/')
  }

  let allChunks: RagChunk[] = []
  let metadata = {
    created_at: '',
    model: ''
  }

  for (const file of partFiles) {
    const partPath = path.join(partsDir, file)
    const part = JSON.parse(fs.readFileSync(partPath, 'utf8'))
    
    if (allChunks.length === 0) {
      metadata = {
        created_at: part.created_at,
        model: part.model
      }
    }
    
    allChunks = allChunks.concat(part.chunks)
  }

  indexCache = { ...metadata, chunks: allChunks }
  indexLoadTime = Date.now()
  
  return indexCache
}

export function getCorpusStats() {
  const index = loadSplitIndex()
  const files = [...new Set(index.chunks.map(c => c.file))].sort()
  
  // Get paper metadata for each file
  const papers = files
    .map(filename => getPaperMetadata(filename))
    .filter((metadata): metadata is PaperMetadata => metadata !== null)
  
  return {
    total_documents: files.length,
    total_chunks: index.chunks.length,
    filenames: files,
    papers: papers,
    index_created: index.created_at,
    cache_age_seconds: indexLoadTime 
      ? Math.floor((Date.now() - indexLoadTime) / 1000) 
      : 0
  }
}

// Clear cache (useful after reindexing)
export function clearIndexCache() {
  indexCache = null
  indexLoadTime = null
}