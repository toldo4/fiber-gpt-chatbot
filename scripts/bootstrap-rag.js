// bootstrap-rag.js — Responses API path (no Assistants)
// Creates a vector store and uploads files from ./docs, then saves vector_store_id.

require('dotenv').config();
const fs = require('fs');
const path = require('path');
const OpenAI = require('openai');

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

function getVectorStores(c) {
  // Newer SDKs use top-level vectorStores; older ones use beta.vectorStores
  if (c.vectorStores?.create) return c.vectorStores;
  if (c.beta?.vectorStores?.create) return c.beta.vectorStores;
  throw new Error('Vector Stores API not found. Try: npm i openai@latest');
}

async function uploadFilesToStore(vs, storeId, filePaths) {
  if (vs.fileBatches?.uploadAndPoll) {
    return vs.fileBatches.uploadAndPoll(storeId, { files: filePaths.map(fp => fs.createReadStream(fp)) });
  }
  if (vs.files?.uploadAndPoll) {
    return vs.files.uploadAndPoll(storeId, { files: filePaths.map(fp => fs.createReadStream(fp)) });
  }
  throw new Error('uploadAndPoll not found on vector stores. Update SDK.');
}

(async () => {
  try {
    const vs = getVectorStores(client);

    // 1) Create vector store
    const store = await vs.create({ name: 'Cranberry-Core' });
    console.log('Vector store created:', store.id);

    // 2) Collect files
    const docsDir = path.join(__dirname, 'docs');
    if (!fs.existsSync(docsDir)) fs.mkdirSync(docsDir);
    const filePaths = fs.readdirSync(docsDir).map(f => path.join(docsDir, f));
    if (filePaths.length === 0) {
      console.log('Put PDFs/CSVs/TXT into ./docs first, then rerun.');
      process.exit(0);
    }

    // 3) Upload files
    await uploadFilesToStore(vs, store.id, filePaths);
    console.log('Uploaded files to vector store:', store.id);

    // 4) Save just the vector_store_id
    fs.writeFileSync(
      path.join(__dirname, 'rag-ids.json'),
      JSON.stringify({ vector_store_id: store.id }, null, 2)
    );
    console.log('Saved vector_store_id to rag-ids.json');
  } catch (e) {
    console.error('Bootstrap error:', e?.response?.data || e);
    process.exit(1);
  }
})();
