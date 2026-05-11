'use client';

import type { ChatRequestOptions, Message } from 'ai';
import cx from 'classnames';
import equal from 'fast-deep-equal';
import { AnimatePresence, motion } from 'framer-motion';
import { memo, useState } from 'react';

import { cn } from '@/lib/utils';
import { PencilEditIcon, SparklesIcon } from './icons';
import { Markdown } from './markdown';
import { MessageActions } from './message-actions';
import { MessageEditor } from './message-editor';
import { Button } from './ui/button';
import { Tooltip, TooltipContent, TooltipTrigger } from './ui/tooltip';
import { Weather } from './weather';

function renumberCitations(text: string): string {
  const sourcesMatch = text.match(/\n(#{1,3}\s+Sources?|Sources?)\s*:/i);
  const splitIdx = sourcesMatch?.index;
  const mainText = splitIdx !== undefined ? text.slice(0, splitIdx) : text;

  const seen = new Map<string, number>();
  let counter = 1;
  for (const [, n] of mainText.matchAll(/\[(\d+)\]/g)) {
    if (!seen.has(n)) seen.set(n, counter++);
  }
  if (seen.size === 0) return text;

  const remap = (s: string) => s.replace(/\[(\d+)\]/g, (_, n) => `[${seen.get(n) ?? n}]`);

  if (splitIdx === undefined) return remap(text);

  // Skip the leading '\n' so lines[0] is the "Sources:" heading, not an empty string
  const lines = text.slice(splitIdx + 1).split('\n');
  const headerLine = lines[0];
  const entries: Array<{ origNum: string; body: string[] }> = [];
  let cur: { origNum: string; body: string[] } | null = null;
  for (const line of lines.slice(1)) {
    const m = line.match(/^\[(\d+)\]/);
    if (m) { if (cur) entries.push(cur); cur = { origNum: m[1], body: [line] }; }
    else if (cur) cur.body.push(line);
  }
  if (cur) entries.push(cur);

  const sorted = entries
    .filter(e => seen.has(e.origNum))
    .sort((a, b) => seen.get(a.origNum)! - seen.get(b.origNum)!);

  // Normalize: strip trailing blank lines from each entry, then add exactly one blank
  // line between entries so Markdown renders each source as a separate paragraph
  const bodyLines: string[] = [];
  for (let i = 0; i < sorted.length; i++) {
    const remapped = sorted[i].body.map(remap);
    while (remapped.length > 1 && remapped[remapped.length - 1].trim() === '') remapped.pop();
    bodyLines.push(...remapped);
    if (i < sorted.length - 1) bodyLines.push('');
  }

  return remap(mainText) + '\n' + [headerLine, ...bodyLines].join('\n');
}

const PurePreviewMessage = ({
  chatId,
  message,
  isLoading,
  setMessages,
  reload,
}: {
  chatId: string;
  message: Message;
  isLoading: boolean;
  setMessages: (
    messages: Message[] | ((messages: Message[]) => Message[]),
  ) => void;
  reload: (
    chatRequestOptions?: ChatRequestOptions,
  ) => Promise<string | null | undefined>;
}) => {
  const [mode, setMode] = useState<'view' | 'edit'>('view');

  const content = typeof message.content === 'string'
    ? renumberCitations(message.content)
    : message.content as string;

  return (
    <AnimatePresence>
      <motion.div
        className="w-full mx-auto max-w-3xl px-4 group/message"
        initial={{ y: 5, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        data-role={message.role}
      >
        <div
          className={cn(
            'flex gap-4 w-full group-data-[role=user]/message:ml-auto group-data-[role=user]/message:max-w-2xl',
            {
              'w-full': mode === 'edit',
              'group-data-[role=user]/message:w-fit': mode !== 'edit',
            },
          )}
        >
          {message.role === 'assistant' && (
            <div className="size-8 flex items-center rounded-full justify-center ring-1 shrink-0 ring-border bg-background">
              <div className="translate-y-px">
                <SparklesIcon size={14} />
              </div>
            </div>
          )}

          <div className="flex flex-col gap-2 w-full">
            {message.content && mode === 'view' && (
              <div className="flex flex-row gap-2 items-start">
                {message.role === 'user' && (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="ghost"
                        className="px-2 h-fit rounded-full text-muted-foreground opacity-0 group-hover/message:opacity-100"
                        onClick={() => {
                          setMode('edit');
                        }}
                      >
                        <PencilEditIcon />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Edit message</TooltipContent>
                  </Tooltip>
                )}

                <div
                  className={cn('flex flex-col gap-4', {
                    'bg-primary text-primary-foreground px-3 py-2 rounded-xl':
                      message.role === 'user',
                  })}
                >
                  {message.role === 'assistant' && (
                    <p>
                      <strong className="text-cutler-green dark:text-cupola-white">
                        This AI-generated response is based exclusively on the curated literature sources incorporated into this chatbot's knowledge base. It does not access or retrieve information from external databases, websites, or general knowledge sources. The opinions, interpretations, findings, conclusions, or recommendations expressed herein are generated by the AI system and do not necessarily reflect the views of the research team members at Ohio University (OU), the USDA, or the Office of Dietary Supplements (ODS) at the NIH.
                      </strong>
                    </p>
                  )}
                  {(() => {
                    const sourcesMatch = content.match(/\n(#{1,3}\s+Sources?|Sources?)\s*:/i);
                    if (sourcesMatch && sourcesMatch.index !== undefined) {
                      const splitIndex = sourcesMatch.index;
                      const mainContent = content.slice(0, splitIndex);
                      const sourcesContent = content.slice(splitIndex);
                      const linkedSources = sourcesContent
                        .replace(/DOI:\s*([\w./\-]+)/g, (_: string, doi: string) => `[DOI: ${doi}](https://doi.org/${doi})`)
                        .replace(/Link:\s*(\S+)/g, (_: string, url: string) => `[Link: ${url}](${url})`);
                      return (
                        <>
                          <Markdown>{mainContent}</Markdown>
                          <div className="text-primary text-sm border-t border-primary/25 pt-2 mt-2">
                            <Markdown>{linkedSources}</Markdown>
                          </div>
                        </>
                      );
                    }
                    return <Markdown>{content}</Markdown>;
                  })()}
                </div>
              </div>
            )}

            {message.content && mode === 'edit' && (
              <div className="flex flex-row gap-2 items-start">
                <div className="size-8" />

                <MessageEditor
                  key={message.id}
                  message={message}
                  setMode={setMode}
                  setMessages={setMessages}
                  reload={reload}
                />
              </div>
            )}

            {message.toolInvocations && message.toolInvocations.length > 0 && (
              <div className="flex flex-col gap-4">
                {message.toolInvocations.map((toolInvocation) => {
                  const { toolName, toolCallId, state, args } = toolInvocation;

                  if (state === 'result') {
                    const { result } = toolInvocation;

                    return (
                      <div key={toolCallId}>
                        {toolName === 'getWeather' ? (
                          <Weather weatherAtLocation={result} />
                        ) : (
                          <pre>{JSON.stringify(result, null, 2)}</pre>
                        )}
                      </div>
                    );
                  }
                  return (
                    <div
                      key={toolCallId}
                      className={cx({
                        skeleton: ['getWeather'].includes(toolName),
                      })}
                    >
                      {toolName === 'getWeather' ? <Weather /> : null}
                    </div>
                  );
                })}
              </div>
            )}

            <MessageActions
              key={`action-${message.id}`}
              chatId={chatId}
              message={message}
              isLoading={isLoading}
            />
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
};

export const PreviewMessage = memo(
  PurePreviewMessage,
  (prevProps, nextProps) => {
    if (prevProps.isLoading !== nextProps.isLoading) return false;
    if (prevProps.message.content !== nextProps.message.content) return false;
    if (
      !equal(
        prevProps.message.toolInvocations,
        nextProps.message.toolInvocations,
      )
    )
      return false;

    return true;
  },
);

export const ThinkingMessage = () => {
  const role = 'assistant';

  return (
    <motion.div
      className="w-full mx-auto max-w-3xl px-4 group/message "
      initial={{ y: 5, opacity: 0 }}
      animate={{ y: 0, opacity: 1, transition: { delay: 1 } }}
      data-role={role}
    >
      <div
        className={cx(
          'flex gap-4 group-data-[role=user]/message:px-3 w-full group-data-[role=user]/message:w-fit group-data-[role=user]/message:ml-auto group-data-[role=user]/message:max-w-2xl group-data-[role=user]/message:py-2 rounded-xl',
          {
            'group-data-[role=user]/message:bg-muted': true,
          },
        )}
      >
        <div className="size-8 flex items-center rounded-full justify-center ring-1 shrink-0 ring-border">
          <SparklesIcon size={14} />
        </div>

        <div className="flex flex-col gap-2 w-full">
          <div className="flex flex-col gap-4 text-muted-foreground">
            Thinking...
          </div>
        </div>
      </div>
    </motion.div>
  );
};
