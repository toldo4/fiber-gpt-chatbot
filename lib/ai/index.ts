// lib/ai/index.ts
import { openai } from '@ai-sdk/openai';
import { google } from '@ai-sdk/google';
import { experimental_wrapLanguageModel as wrapLanguageModel } from 'ai';

import { customMiddleware } from './custom-middleware';
import { models } from './models';

export const customModel = (apiIdentifier: string) => {
  // Find the model configuration based on the identifier
  const modelConfig = models.find(m => m.apiIdentifier === apiIdentifier || m.id === apiIdentifier);
  
  // Default to OpenAI if not explicitly marked as google
  const isGoogle = modelConfig?.provider === 'google';
  const baseModel = isGoogle ? google(apiIdentifier) : openai(apiIdentifier);

  return wrapLanguageModel({
    model: baseModel,
    middleware: customMiddleware,
  });
};

export const imageGenerationModel = openai.image('dall-e-3');