// Define your models here.

export interface Model {
  id: string;
  label: string;
  apiIdentifier: string;
  description: string;
  provider: 'openai' | 'google';
  temperature?: number;
}

export const models: Array<Model> = [
  {
    id: 'gpt-4.1',
    label: 'GPT 4.1',
    apiIdentifier: 'gpt-4.1',
    description: '',
    provider: 'openai',
  },
    {
    id: 'gpt-4o-mini',
    label: 'GPT 4o mini',
    apiIdentifier: 'gpt-4o-mini',
    description: 'Small model for fast, lightweight tasks',
    provider: 'openai',
  },
  {
    id: 'gpt-4o',
    label: 'GPT 4o',
    apiIdentifier: 'gpt-4o',
    description: 'For complex, multi-step tasks',
    provider: 'openai',
  },
// {
//     id: 'gemini-2.5-pro',
//     label: 'Gemini 2.5 Pro',
//     apiIdentifier: 'gemini-2.5-pro', // Updated to 2.5
//     description: 'Complex reasoning tasks',
//     provider: 'google',
//   },
//   {
//     id: 'gemini-2.5-flash',
//     label: 'Gemini 2.5 Flash',
//     apiIdentifier: 'gemini-2.5-flash', // Updated to 2.5
//     description: 'Fast and versatile performance',
//     provider: 'google',
//   },
  {
    id: 'gpt-5.5-2026-04-23',
    label: 'GPT 5.5',
    apiIdentifier: 'gpt-5.5-2026-04-23',
    description: '',
    provider: 'openai',
    temperature: 1,
  },
  {
    id: 'gpt-5.1',
    label: 'GPT 5.1',
    apiIdentifier: 'gpt-5.1',
    description: 'Latest OpenAI model with improved context handling and accuracy.',
    provider: 'openai',
  },
  {
    id: 'gemini-3.1-pro',
    label: 'Gemini 3.1 Pro',
    apiIdentifier: 'gemini-3.1-pro-preview',
    description: 'Advanced intelligence with complex problem-solving skills and powerful reasoning capabilities.',
    provider: 'google',
  },
  {
    id: 'gemini-3.1-flash-lite',
    label: 'Gemini 3.1 Flash Lite',
    apiIdentifier: 'gemini-3.1-flash-lite-preview',
    description: 'Fast and budget-friendly multimodal model with frontier-class performance.',
    provider: 'google',
  },
] as const;

export const DEFAULT_MODEL_NAME: string = 'gpt-5.1';
