export interface PaperMetadata {
  filename: string;
  title: string;
  journal: string;
  year: string;
  doi: string;
}

export const PAPER_METADATA: Record<string, PaperMetadata> = {
  "2404.16130v2.pdf": {
    filename: "2404.16130v2.pdf",
    title: "From Local to Global: A GraphRAG Approach to Query-Focused Summarization",
    journal: "arXiv preprint",
    year: "2025",
    doi: "10.48550/arXiv.2404.16130"
  },
  "2409.15355v5.pdf": {
    filename: "2409.15355v5.pdf",
    title: "Block-Attention for Efficient Prefilling",
    journal: "International Conference on Learning Representations (ICLR 2025)",
    year: "2025",
    doi: "10.48550/arXiv.2409.15355"
  },
  "2411.17116v3.pdf": {
    filename: "2411.17116v3.pdf",
    title: "Star Attention: Efficient LLM Inference over Long Sequences",
    journal: "International Conference on Machine Learning (ICML 2025)",
    year: "2025",
    doi: "10.48550/arXiv.2411.17116"
  },
  "2501.06713v3.pdf": {
    filename: "2501.06713v3.pdf",
    title: "MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation",
    journal: "arXiv preprint",
    year: "2025",
    doi: "10.48550/arXiv.2501.06713"
  }
};

export function getPaperMetadata(filename: string): PaperMetadata | null {
  return PAPER_METADATA[filename] || null;
}

export function formatPaperCitation(metadata: PaperMetadata): string {
  return `${metadata.title}\nJournal: ${metadata.journal} (${metadata.year})\nDOI: ${metadata.doi}`;
}