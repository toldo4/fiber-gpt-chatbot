// build-index.js — Local RAG indexer with automatic splitting
require('dotenv').config();
const fs = require('fs');
const path = require('path');
const OpenAI = require('openai');
const { globSync } = require('glob');

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const PARTS = 5; // Number of parts to split into

/** Try to load pdf-parse in a way that works across CJS/ESM variants. */
let pdfParse = null;
try {
  const mod = require('pdf-parse');
  pdfParse = typeof mod === 'function' ? mod : (mod && mod.default ? mod.default : null);
} catch (e) {
  // not installed or load failed — we'll just skip PDFs gracefully
}

/** Robust chunker: guarantees forward progress and clamps overlap */
function chunkText(text, chunkSize = 1200, overlap = 150) {
  if (!text || !text.trim()) return [];
  const clean = text.replace(/\s+/g, ' ').trim();
  overlap = Math.max(0, Math.min(overlap, Math.floor(chunkSize / 2)));
  const chunks = [];
  let start = 0;

  while (start < clean.length) {
    let end = Math.min(clean.length, start + chunkSize);
    let slice = clean.slice(start, end);

    if (end < clean.length) {
      const punct = Math.max(slice.lastIndexOf('. '), slice.lastIndexOf('? '), slice.lastIndexOf('! '));
      if (punct > Math.floor(chunkSize * 0.6)) {
        slice = slice.slice(0, punct + 1);
        end = start + slice.length;
      }
    }

    if (slice.length > 0) chunks.push(slice);
    if (end >= clean.length) break;
    const step = Math.max(1, (end - start) - overlap);
    start = start + step;
  }

  return chunks;
}

/** Load docling-parsed records from parsed-docs.json, grouped by file+section */
function loadDoclingChunks() {
  const parsedPath = path.join(__dirname, 'parsed-docs.json');
  if (!fs.existsSync(parsedPath)) return null;

  const records = JSON.parse(fs.readFileSync(parsedPath, 'utf8'));
  // Group text blocks by file → section
  const grouped = {};
  for (const { file, section, text } of records) {
    if (!grouped[file]) grouped[file] = {};
    if (!grouped[file][section]) grouped[file][section] = [];
    grouped[file][section].push(text);
  }

  const chunks = [];
  for (const file of Object.keys(grouped).sort()) {
    for (const section of Object.keys(grouped[file])) {
      const combined = grouped[file][section].join(' ');
      const textChunks = chunkText(combined);
      textChunks.forEach((c, idx) => {
        chunks.push({
          id: `${file}#${section}#${idx}`,
          file,
          section,
          // Prepend section name to the text used for embedding so retrieval
          // benefits from the structural context.
          text: c,
          embedText: `[${section}] ${c}`,
        });
      });
    }
    console.log(`  Prepared chunks from ${file} (via docling)`);
  }
  return chunks;
}

/** Read file text for .pdf (if pdf-parse available) and .txt */
async function readFileText(fp) {
  const ext = path.extname(fp).toLowerCase();
  if (ext === '.pdf') {
    if (!pdfParse) {
      console.warn(`⚠️  pdf-parse not available; skipping PDF: ${path.basename(fp)}`);
      return '';
    }
    const buf = fs.readFileSync(fp);
    const data = await pdfParse(buf);
    return (data && data.text) ? data.text : '';
  }
  if (ext === '.txt') {
    return fs.readFileSync(fp, 'utf8');
  }
  return '';
}

/** Create embeddings in batches */
async function embedBatch(texts) {
  const res = await client.embeddings.create({
    model: 'text-embedding-3-small',
    input: texts
  });
  return res.data.map(d => d.embedding);
}

(async () => {
  try {
    const docsDir = path.join(__dirname, 'docs');
    if (!fs.existsSync(docsDir)) {
      console.log('Create a ./docs folder and add PDFs or TXTs, then rerun.');
      process.exit(0);
    }

    const files = globSync(path.join(docsDir, '**/*.@(pdf|txt)').replace(/\\/g, '/'));
    if (files.length === 0) {
      console.log('No PDF/TXT found in ./docs. Add files and rerun.');
      process.exit(0);
    }

    // Prefer docling-parsed data (section-aware) over raw pdf-parse
    let allChunks = loadDoclingChunks();

    if (allChunks) {
      console.log(`Loaded ${allChunks.length} section-aware chunks from parsed-docs.json`);
    } else {
      console.log('No parsed-docs.json found — falling back to pdf-parse');
      allChunks = [];
      for (const fp of files) {
        const base = path.basename(fp);
        const raw = await readFileText(fp);
        if (!raw || !raw.trim()) {
          console.warn(`(empty or unreadable) ${base} — skipped`);
          continue;
        }
        const chunks = chunkText(raw);
        chunks.forEach((c, idx) => {
          allChunks.push({ id: `${base}#${idx}`, file: base, text: c });
        });
        console.log(`Prepared ${chunks.length} chunks from ${base}`);
      }
    }

    if (allChunks.length === 0) {
      console.log('No text extracted from any file. Check your docs.');
      process.exit(0);
    }

    // Embed chunks in batches (use embedText if present, else text)
    const B = 50;
    for (let i = 0; i < allChunks.length; i += B) {
      const batch = allChunks.slice(i, i + B);
      const vecs = await embedBatch(batch.map(b => b.embedText ?? b.text));
      batch.forEach((b, j) => {
        b.embedding = vecs[j];
        delete b.embedText; // don't store the augmented text in the index
      });
      console.log(`Embedded ${Math.min(i + B, allChunks.length)}/${allChunks.length}`);
    }

    // Create parts directory
    const partsDir = path.join(__dirname, '../index-parts');
    if (!fs.existsSync(partsDir)) {
      fs.mkdirSync(partsDir);
    }

    // Split and save into parts
    const totalChunks = allChunks.length;
    const chunkSize = Math.ceil(totalChunks / PARTS);
    const timestamp = new Date().toISOString();

    console.log(`\nSplitting ${totalChunks} chunks into ${PARTS} parts...`);

    for (let i = 0; i < PARTS; i++) {
      const start = i * chunkSize;
      const end = Math.min(start + chunkSize, totalChunks);
      const partChunks = allChunks.slice(start, end);
      
      const part = {
        part: i + 1,
        total_parts: PARTS,
        created_at: timestamp,
        model: 'text-embedding-3-small',
        chunks: partChunks
      };
      
      const partPath = path.join(partsDir, `index-part-${i + 1}.json`);
      fs.writeFileSync(partPath, JSON.stringify(part));
      
      const sizeMB = (fs.statSync(partPath).size / (1024 * 1024)).toFixed(2);
      console.log(` Part ${i + 1}/${PARTS}: ${partChunks.length} chunks (${sizeMB} MB)`);
    }

    console.log(' Built split index in ./index-parts/');
  } catch (e) {
    console.error('Index error:', e?.response?.data || e);
    process.exit(1);
  }
})();