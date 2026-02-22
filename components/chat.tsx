'use client';

import { useChat } from 'ai/react';

import { ChatHeader } from '@/components/chat-header';
import { Messages } from './messages';
import { MultimodalInput } from './multimodal-input';
import { SuggestedActions } from './suggested-actions';

export function Chat({
  id,
  selectedModelId,
}: {
  id: string;
  selectedModelId: string;
}) {
  const {
    messages,
    setMessages,
    handleSubmit,
    input,
    setInput,
    append,
    isLoading,
    stop,
    reload,
  } = useChat({
    id,
    body: { id, modelId: selectedModelId },
    experimental_throttle: 100,
  });

  // Check if the last message was from the assistant and we aren't loading
  const isLastMessageAssistant = 
    messages.length > 0 && 
    messages[messages.length - 1].role === 'assistant' && 
    !isLoading;

  return (
    <div className="flex flex-col min-w-0 h-dvh bg-background">
      <ChatHeader selectedModelId={selectedModelId} />

      <Messages
        chatId={id}
        isLoading={isLoading}
        messages={messages}
        setMessages={setMessages}
        reload={reload}
        append={append}
      />

      {(messages.length === 0 || isLastMessageAssistant) && (
        <div className="flex mx-auto px-4 bg-background pb-4 md:pb-6 gap-2 w-full md:max-w-3xl">
          <SuggestedActions 
            chatId={id} 
            append={append} 
            isFollowUp={isLastMessageAssistant} 
          />
        </div>
      )}

      <form className="flex mx-auto px-4 bg-background pb-4 md:pb-6 gap-2 w-full md:max-w-3xl">
        <MultimodalInput
          chatId={id}
          input={input}
          setInput={setInput}
          handleSubmit={handleSubmit}
          isLoading={isLoading}
          stop={stop}
          messages={messages}
          setMessages={setMessages}
          append={append}
        />
      </form>
    </div>
  );
}