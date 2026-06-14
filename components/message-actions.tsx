import type { Message } from 'ai';
import { memo, useState } from 'react';
import { toast } from 'sonner';
import { useCopyToClipboard } from 'usehooks-ts';

import { CopyIcon, ThumbDownIcon, ThumbUpIcon } from './icons';
import { Button } from './ui/button';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from './ui/tooltip';

export function PureMessageActions({
  chatId,
  message,
  isLoading,
}: {
  chatId: string;
  message: Message;
  isLoading: boolean;
}) {
  const [_, copyToClipboard] = useCopyToClipboard();
  const [rating, setRating] = useState<'up' | 'down' | null>(null);

  if (isLoading) return null;
  if (message.role === 'user') return null;
  if (message.toolInvocations && message.toolInvocations.length > 0)
    return null;

  const sendFeedback = async (value: 'up' | 'down') => {
    const next = rating === value ? null : value;
    setRating(next);
    if (!next) return;

    const messagePreview =
      typeof message.content === 'string'
        ? message.content.slice(0, 200)
        : '';

    await fetch('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messageId: message.id, chatId, rating: next, messagePreview }),
    });

    toast.success(next === 'up' ? 'Thanks for the feedback!' : 'Thanks, we\'ll work to improve.');
  };

  return (
    <TooltipProvider delayDuration={0}>
      <div className="flex flex-row gap-2">
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              className="py-1 px-2 h-fit text-muted-foreground"
              variant="outline"
              onClick={async () => {
                await copyToClipboard(message.content);
                toast.success('Copied to clipboard!');
              }}
            >
              <CopyIcon />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Copy</TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              className={`py-1 px-2 h-fit ${rating === 'up' ? 'text-green-600 border-green-600' : 'text-muted-foreground'}`}
              variant="outline"
              onClick={() => sendFeedback('up')}
            >
              <ThumbUpIcon />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Helpful</TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              className={`py-1 px-2 h-fit ${rating === 'down' ? 'text-red-500 border-red-500' : 'text-muted-foreground'}`}
              variant="outline"
              onClick={() => sendFeedback('down')}
            >
              <ThumbDownIcon />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Not helpful</TooltipContent>
        </Tooltip>
      </div>
    </TooltipProvider>
  );
}

export const MessageActions = memo(
  PureMessageActions,
  (prevProps, nextProps) => {
    if (prevProps.isLoading !== nextProps.isLoading) return false;

    return true;
  },
);
