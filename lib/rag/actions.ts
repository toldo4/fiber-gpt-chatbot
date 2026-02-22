'use server';

import { getCorpusStats as getStats } from './index-loader';

export async function getCorpusStats() {
  const stats = getStats();
  return {
    files: stats.filenames,
  };
}