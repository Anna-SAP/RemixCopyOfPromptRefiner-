import { GoogleGenAI, Type } from "@google/genai";
import { AttachedImage } from "../types";

const SYSTEM_INSTRUCTION = `
You are a cutting-edge Prompt Engineer for modern, highly intelligent Large Language Models.
Your goal is to take the user's raw, unstructured request (which may include images) and refine it into a "low-restriction, high-focus" modern optimized version.

**Core Philosophy:**
Current LLMs possess advanced reasoning and intelligence. Lengthy, overly detailed, and micro-managed prompts stifle their logic and creativity. We need a new paradigm: reduce restrictions, focus on the core, and let the model determine the optimal execution path.

**Optimization Guidelines:**
1. **Remove Fluff:** Strip away meaningless background noise, over-explanations, and obvious common sense for a highly intelligent model.
2. **Stop Micro-managing:** Clearly define the "What" (the goal) and give back the "How" (the execution path) to the model. Delete overly granular step-by-step instructions unless it's a strict machine SOP.
3. **Focus on the Core:** Distill the ultimate objective, the core audience, and the non-negotiable constraints.
4. **Keep it Open:** Avoid rigid word counts, specific vocabulary limits, or overly strict formatting requirements unless explicitly demanded by a system integration, giving the model space to utilize its intelligence.

**Style & Tone:**
Professional, concise, straight to the point, minimalist.
Objective, clear, efficient, empowering.

**Target Audience:**
A state-of-the-art Large Language Model (the one that will receive this optimized prompt).

**Language STRICT RULE:**
The output language must strictly match the user's input language.
- If the user provides input in **Simplified Chinese**, the entire response MUST be in **Simplified Chinese**.
- If the user provides input in **English**, the entire response MUST be in **English**.

**Output Format:**
- **Analysis:** A brief 1-2 sentence explanation of what was stripped away and what was focused on.
- **Refined Prompt:** The actual optimized prompt in a code block. Keep the refined prompt as clean and unrestrictive as possible, focusing purely on context, objective, and strict constraints.
`;

export class GeminiService {
  private ai: GoogleGenAI;

  constructor() {
    // API key is injected via process.env.API_KEY
    this.ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  }

  /**
   * Refines the prompt using Gemini 3.1 Pro
   */
  async refinePromptStream(
    inputText: string, 
    images: AttachedImage[], 
    onChunk: (text: string) => void
  ): Promise<void> {
    
    const model = 'gemini-3.1-pro-preview';

    const parts: any[] = [];

    // Add images if available
    if (images.length > 0) {
      images.forEach(img => {
        // Strip data:image/xyz;base64, prefix if present for the API call
        const base64Data = img.data.split(',')[1];
        parts.push({
          inlineData: {
            mimeType: img.mimeType,
            data: base64Data
          }
        });
      });
    }

    // Add text prompt
    parts.push({
      text: `User Request: "${inputText}". \n\nPlease refine this into a professional prompt.`
    });

    try {
      const responseStream = await this.ai.models.generateContentStream({
        model: model,
        contents: {
          role: 'user',
          parts: parts
        },
        config: {
          systemInstruction: SYSTEM_INSTRUCTION,
          temperature: 0.7,
        }
      });

      for await (const chunk of responseStream) {
        if (chunk.text) {
          onChunk(chunk.text);
        }
      }
    } catch (error) {
      console.error("Gemini API Error:", error);
      throw error;
    }
  }
}

export const geminiService = new GeminiService();