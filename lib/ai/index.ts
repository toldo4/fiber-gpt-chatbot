import { openai } from '@ai-sdk/openai';
import { google } from '@ai-sdk/google';
import { experimental_wrapLanguageModel as wrapLanguageModel, type LanguageModelV1 } from 'ai';

import { customMiddleware } from './custom-middleware';
import { models } from './models';

export const customModel = (apiIdentifier: string) => {
  const modelConfig = models.find(m => m.apiIdentifier === apiIdentifier || m.id === apiIdentifier);
  
  const isGoogle = modelConfig?.provider === 'google';
  const baseModel = isGoogle ? google(apiIdentifier) : openai(apiIdentifier);

  return wrapLanguageModel({
    model: baseModel as unknown as LanguageModelV1,
    middleware: customMiddleware,
  });
};

export const imageGenerationModel = openai.image('dall-e-3');