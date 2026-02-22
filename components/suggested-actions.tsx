'use client';

import { ChatRequestOptions, CreateMessage, Message } from 'ai';
import { motion } from 'framer-motion';
import { memo } from 'react';

import { Button } from './ui/button';

interface SuggestedActionsProps {
  chatId: string;
  append: (
    message: Message | CreateMessage,
    chatRequestOptions?: ChatRequestOptions,
  ) => Promise<string | null | undefined>;
  isFollowUp?: boolean;
}

function PureSuggestedActions({ chatId, append, isFollowUp }: SuggestedActionsProps) {
const initialActions = [
  {
    title: 'What is the Codex Alimentarius',
    label: 'definition of dietary fiber?',
    action: 'What is the Codex Alimentarius definition of dietary fiber?',
  },
  {
    title: 'What are the differences between',
    label: 'soluble and insoluble dietary fiber?',
    action: 'What are the differences between soluble and insoluble dietary fiber?',
  },
  {
    title: 'What AOAC methods are used',
    label: 'to measure total dietary fiber?',
    action: 'What AOAC methods are used to measure total dietary fiber?',
  },
  {
    title: 'What is resistant starch and',
    label: 'how is it measured in foods?',
    action: 'What is resistant starch and how is it measured in foods?',
  },
];

  // The two follow-up actions shown after a response
  const followUpActions: Array<{ title: string; label: string; action: string }> = [
    // {
    //   title: 'Tell me more',
    //   label: ' about the research methodology.',
    //   action: 'Can you tell me more about the research methodology used in these studies?',
    // },
    // {
    //   title: 'Are there any',
    //   label: ' safety concerns or interactions?',
    //   action: 'What are the safety concerns or potential drug interactions for these supplements?',
    // },
  ];

  const actions = isFollowUp ? followUpActions : initialActions;

  // Don't render if follow-up actions are empty
  if (isFollowUp && followUpActions.length === 0) {
    return null;
  }

  return (
    <div className="grid sm:grid-cols-2 gap-2 w-full">
      {actions.map((suggestedAction, index) => (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 20 }}
          transition={{ delay: 0.05 * index }}
          key={`${isFollowUp ? 'followup' : 'initial'}-${index}`}
          className="block"
        >
          <Button
            variant="ghost"
            onClick={async () => {
              window.history.replaceState({}, '', `/chat/${chatId}`);

              append({
                role: 'user',
                content: suggestedAction.action,
              });
            }}
            className="text-left border rounded-xl px-4 py-3.5 text-sm flex-1 gap-1 sm:flex-col w-full h-auto justify-start items-start"
          >
            <span className="font-medium">{suggestedAction.title}</span>
            <span className="text-muted-foreground">
              {suggestedAction.label}
            </span>
          </Button>
        </motion.div>
      ))}
    </div>
  );
}

export const SuggestedActions = memo(
  PureSuggestedActions, 
  (prev, next) => prev.isFollowUp === next.isFollowUp && prev.chatId === next.chatId
);