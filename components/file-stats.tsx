'use client';

import { memo, useEffect, useState } from 'react';

import { FileIcon } from '@/components/icons';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
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
}

function PureFileStats() {
  const [papers, setPapers] = useState<PaperInfo[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/rag-stats')
      .then((response) => response.json())
      .then((data) => {
        if (data && Array.isArray(data.papers)) {
          setPapers(data.papers);
        }
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const handlePaperClick = (doi: string) => {
    window.open(`https://doi.org/${doi}`, '_blank');
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="icon" className="w-9 h-9">
          <FileIcon />
          <span className="sr-only">View Research Papers</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-[500px]">
        <DropdownMenuLabel>
          Available Research Papers ({papers.length})
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <div className="max-h-[400px] overflow-y-auto">
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
              <DropdownMenuItem
                key={index}
                className="cursor-pointer flex-col items-start gap-1 py-3 px-3 hover:bg-accent"
                onClick={() => handlePaperClick(paper.doi)}
              >
                <div className="flex items-start gap-2 w-full">
                  <div className="shrink-0 mt-0.5">
                    <FileIcon size={14} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-sm leading-tight mb-1 text-foreground hover:text-primary">
                      {paper.title}
                    </div>
                    <div className="text-xs text-muted-foreground space-y-0.5">
                      <div>
                        <span className="font-medium">Journal:</span>{' '}
                        {paper.journal} ({paper.year})
                      </div>
                      <div>
                        <span className="font-medium">DOI:</span>{' '}
                        <span className="font-mono">{paper.doi}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </DropdownMenuItem>
            ))
          )}
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export const FileStats = memo(PureFileStats);