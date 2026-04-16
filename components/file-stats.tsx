'use client';

import { memo, useEffect, useState } from 'react';

import { FileIcon } from '@/components/icons';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

interface PaperInfo {
  filename: string;
  title: string;
  journal: string;
  year: string;
  doi: string;
  link?: string;
}

interface SectionInfo {
  name: string;
  blocks: number;
}

interface PaperStats {
  filename: string;
  totalBlocks: number;
  sections: SectionInfo[];
}

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 12 12"
      fill="none"
      className={`shrink-0 transition-transform duration-150 ${open ? 'rotate-90' : ''}`}
    >
      <path d="M4 2l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function PaperRow({
  paper,
  stats,
  onOpen,
}: {
  paper: PaperInfo;
  stats: PaperStats | undefined;
  onOpen: (paper: PaperInfo) => void;
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="border-b border-border/50 last:border-0">
      {/* Paper header row */}
      <div
        className="flex items-start gap-2 py-3 px-3 hover:bg-accent cursor-pointer select-none"
        onClick={() => setExpanded((v) => !v)}
      >
        <div className="shrink-0 mt-1 text-muted-foreground">
          <ChevronIcon open={expanded} />
        </div>
        <div className="shrink-0 mt-0.5">
          <FileIcon size={14} />
        </div>
        <div className="flex-1 min-w-0">
          <div
            className="font-medium text-sm leading-tight mb-1 text-foreground hover:text-primary"
            onClick={(e) => {
              e.stopPropagation();
              onOpen(paper);
            }}
          >
            {paper.title}
          </div>
          <div className="text-xs text-muted-foreground space-y-0.5">
            <div>
              <span className="font-medium">Journal:</span>{' '}
              {paper.journal} ({paper.year})
            </div>
            <div>
              <span className="font-medium">{paper.doi ? 'DOI' : 'Link'}:</span>{' '}
              <span className="font-mono">{paper.doi || paper.link || 'N/A'}</span>
            </div>
            {stats && (
              <div className="text-muted-foreground/70">
                {stats.sections.length} sections · {stats.totalBlocks} blocks
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Collapsible sections */}
      {expanded && stats && (
        <div className="bg-muted/30 px-3 pb-2">
          <div className="ml-6 border-l border-border/60 pl-3 space-y-0.5">
            {stats.sections.map((sec, i) => (
              <div key={i} className="flex items-baseline justify-between gap-2 py-0.5">
                <span className="text-xs text-muted-foreground truncate">{sec.name}</span>
                <span className="text-xs text-muted-foreground/60 shrink-0 tabular-nums">
                  {sec.blocks} blk
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function PureFileStats() {
  const [papers, setPapers] = useState<PaperInfo[]>([]);
  const [paperStats, setPaperStats] = useState<PaperStats[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/rag-stats')
      .then((r) => r.json())
      .then((data) => {
        if (data && Array.isArray(data.papers)) setPapers(data.papers);
        if (data && Array.isArray(data.paperStats)) setPaperStats(data.paperStats);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const handlePaperOpen = (paper: PaperInfo) => {
    if (paper.doi) {
      window.open(`https://doi.org/${paper.doi}`, '_blank');
    } else if (paper.link) {
      window.open(paper.link, '_blank');
    }
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="icon" className="w-9 h-9">
          <FileIcon />
          <span className="sr-only">View Research Papers</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-[520px]">
        <DropdownMenuLabel>
          Available Research Papers ({papers.length})
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <div className="max-h-[500px] overflow-y-auto">
          {loading ? (
            <div className="p-4 text-sm text-muted-foreground text-center">
              Loading papers...
            </div>
          ) : papers.length === 0 ? (
            <div className="p-4 text-sm text-muted-foreground text-center">
              No papers available
            </div>
          ) : (
            papers.map((paper, index) => (
              <PaperRow
                key={index}
                paper={paper}
                stats={paperStats.find((s) => s.filename === paper.filename)}
                onOpen={handlePaperOpen}
              />
            ))
          )}
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export const FileStats = memo(PureFileStats);