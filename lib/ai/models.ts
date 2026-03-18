// Define your models here.

export interface Model {
  id: string;
  label: string;
  apiIdentifier: string;
  description: string;
  provider: 'openai' | 'google';
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
    id: 'gpt-5.1',
    label: 'GPT 5.1',
    apiIdentifier: 'gpt-5.1',
    description: 'Latest OpenAI model with improved context handling and accuracy.',
    provider: 'openai',
  }

] as const;

export const DEFAULT_MODEL_NAME: string = 'gpt-5.1';
