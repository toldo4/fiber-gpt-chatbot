import Link from 'next/link';
import { memo } from 'react';
import ReactMarkdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { SKIP, visit } from 'unist-util-visit';
import { CodeBlock } from './code-block';

const components: Components = {
  code: ({ node, className, children, ...props }) => {
    return (
      <CodeBlock {...props} inline={false}>
        {children}
      </CodeBlock>
    );
  },
  pre: ({ children }) => <>{children}</>,
  ol: ({ node, children, ...props }) => {
    return (
      <ol className="list-decimal list-outside ml-4" {...props}>
        {children}
      </ol>
    );
  },
  li: ({ node, children, ...props }) => {
    return (
      <li className="py-1" {...props}>
        {children}
      </li>
    );
  },
  ul: ({ node, children, ...props }) => {
    return (
      <ul className="list-decimal list-outside ml-4" {...props}>
        {children}
      </ul>
    );
  },
  strong: ({ node, children, ...props }) => {
    return (
      <span className="font-semibold" {...props}>
        {children}
      </span>
    );
  },
  a: ({ node, children, href, ...props }) => {
    return (
      <Link
        className="text-blue-500 hover:underline"
        target="_blank"
        rel="noreferrer"
        href={href ?? '#'}
        {...props}
      >
        {children}
      </Link>
    );
  },
  h1: ({ node, children, ...props }) => {
    return (
      <h1 className="text-3xl font-semibold mt-6 mb-2" {...props}>
        {children}
      </h1>
    );
  },
  h2: ({ node, children, ...props }) => {
    return (
      <h2 className="text-2xl font-semibold mt-6 mb-2" {...props}>
        {children}
      </h2>
    );
  },
  h3: ({ node, children, ...props }) => {
    return (
      <h3 className="text-xl font-semibold mt-6 mb-2" {...props}>
        {children}
      </h3>
    );
  },
  h4: ({ node, children, ...props }) => {
    return (
      <h4 className="text-lg font-semibold mt-6 mb-2" {...props}>
        {children}
      </h4>
    );
  },
  h5: ({ node, children, ...props }) => {
    return (
      <h5 className="text-base font-semibold mt-6 mb-2" {...props}>
        {children}
      </h5>
    );
  },
  h6: ({ node, children, ...props }) => {
    return (
      <h6 className="text-sm font-semibold mt-6 mb-2" {...props}>
        {children}
      </h6>
    );
  },
};

// Matches [Section][N] (e.g. [Methods][2]) or standalone [N] inline citations
const CITATION_RE = /(\[\d+\]\[[^\]]+\]|\[\d+\])/;
const NUM_SECTION_RE = /^(\[\d+\])(\[[^\]]+\])$/;

function citationNodes(match: string) {
  const parts = match.match(NUM_SECTION_RE);
  if (parts) {
    return [
      { type: 'element', tagName: 'span', properties: { className: ['ohio-citation-num'] }, children: [{ type: 'text', value: parts[1] }] },
      { type: 'element', tagName: 'span', properties: { className: ['ohio-citation-text'] }, children: [{ type: 'text', value: parts[2] }] },
    ];
  }
  return [{ type: 'element', tagName: 'span', properties: { className: ['ohio-citation-num'] }, children: [{ type: 'text', value: match }] }];
}

const rehypeCitations = () => (tree: any) => {
  visit(tree, 'text', (node: any, index: number | undefined, parent: any) => {
    if (index == null || !parent) return;
    if (parent.properties?.className?.some((c: string) => c.startsWith('ohio-citation'))) return;

    const parts = node.value.split(CITATION_RE);
    if (parts.length <= 1) return;

    const children = parts.flatMap((part: string) => {
      if (!part) return [];
      if (CITATION_RE.test(part)) return citationNodes(part);
      return [{ type: 'text', value: part }];
    });

    parent.children.splice(index, 1, ...children);
    return [SKIP, index + children.length] as any;
  });
};

const remarkPlugins = [remarkGfm];
const rehypePlugins = [rehypeCitations];

const NonMemoizedMarkdown = ({ children }: { children: string }) => {
  return (
    <ReactMarkdown remarkPlugins={remarkPlugins} rehypePlugins={rehypePlugins} components={components}>
      {children}
    </ReactMarkdown>
  );
};

export const Markdown = memo(
  NonMemoizedMarkdown,
  (prevProps, nextProps) => prevProps.children === nextProps.children,
);
