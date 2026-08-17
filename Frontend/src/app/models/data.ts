export interface CuratedModel {
  name: string;
  description: string;
  params: string;
  reqRamGB: number;
}

export const CURATED_MODELS: CuratedModel[] = [
  // Llama Family
  { name: "llama3.2:latest", description: "Meta's highly capable 3B model. Fast and precise.", params: "3B", reqRamGB: 4 },
  { name: "llama3.2:1b", description: "Meta's ultra-lightweight 1B model for constrained devices.", params: "1B", reqRamGB: 2 },
  { name: "llama3.1:8b", description: "Meta's extremely popular 8B model.", params: "8B", reqRamGB: 7.5 },
  { name: "llama3.1:70b", description: "Meta's massive 70B model. Requires high-end hardware.", params: "70B", reqRamGB: 40 },
  { name: "llama3:8b", description: "Meta's previous generation 8B model.", params: "8B", reqRamGB: 7 },
  { name: "llama3:70b", description: "Meta's previous generation 70B model.", params: "70B", reqRamGB: 40 },
  
  // Mistral Family
  { name: "mistral:latest", description: "The 7B model that sets the standard for open-source AI.", params: "7B", reqRamGB: 7 },
  { name: "mistral-nemo:latest", description: "Mistral's 12B model built in collaboration with Nvidia.", params: "12B", reqRamGB: 10 },
  { name: "mixtral:8x7b", description: "Mistral's powerful Mixture of Experts model.", params: "47B", reqRamGB: 28 },
  { name: "mixtral:8x22b", description: "Mistral's massive Mixture of Experts model.", params: "141B", reqRamGB: 80 },
  
  // Google Gemma Family
  { name: "gemma2:2b", description: "Google's lightweight 2B model.", params: "2B", reqRamGB: 3 },
  { name: "gemma2:9b", description: "Google's powerful 9B model built from the same tech as Gemini.", params: "9B", reqRamGB: 8.5 },
  { name: "gemma2:27b", description: "Google's massive 27B model.", params: "27B", reqRamGB: 20 },
  { name: "gemma:7b", description: "Google's first generation 7B model.", params: "7B", reqRamGB: 7 },
  { name: "gemma:2b", description: "Google's first generation 2B model.", params: "2B", reqRamGB: 3 },
  
  // Qwen Family (Alibaba)
  { name: "qwen2.5:0.5b", description: "Alibaba's ultra-lightweight 0.5B model.", params: "500M", reqRamGB: 1.5 },
  { name: "qwen2.5:1.5b", description: "Alibaba's fast 1.5B model.", params: "1.5B", reqRamGB: 3 },
  { name: "qwen2.5:3b", description: "Alibaba's capable 3B model.", params: "3B", reqRamGB: 4 },
  { name: "qwen2.5:7b", description: "Alibaba's advanced 7B model. Extremely competent at coding.", params: "7B", reqRamGB: 7 },
  { name: "qwen2.5:14b", description: "Alibaba's high-performance 14B model.", params: "14B", reqRamGB: 12 },
  { name: "qwen2.5:32b", description: "Alibaba's powerful 32B model.", params: "32B", reqRamGB: 24 },
  { name: "qwen2.5:72b", description: "Alibaba's massive 72B model.", params: "72B", reqRamGB: 45 },
  
  // Phi Family (Microsoft)
  { name: "phi3:mini", description: "Microsoft's lightweight 3.8B model. Unbelievably fast.", params: "3.8B", reqRamGB: 5 },
  { name: "phi3:medium", description: "Microsoft's highly capable 14B model.", params: "14B", reqRamGB: 12 },
  { name: "phi3.5:latest", description: "Microsoft's updated lightweight model.", params: "3.8B", reqRamGB: 5 },
  
  // Coding Models
  { name: "deepseek-coder-v2:latest", description: "Top-tier coding model by DeepSeek.", params: "16B", reqRamGB: 14 },
  { name: "codellama:7b", description: "Meta's coding-focused variant of Llama 2.", params: "7B", reqRamGB: 7 },
  { name: "codellama:13b", description: "Meta's 13B coding model.", params: "13B", reqRamGB: 10 },
  { name: "codellama:34b", description: "Meta's 34B coding model.", params: "34B", reqRamGB: 22 },
  { name: "qwen2.5-coder:7b", description: "Alibaba's coding-specialized Qwen model.", params: "7B", reqRamGB: 7 },
  { name: "starcoder2:3b", description: "BigCode's lightweight coding model.", params: "3B", reqRamGB: 4 },
  { name: "starcoder2:7b", description: "BigCode's standard coding model.", params: "7B", reqRamGB: 7 },
  { name: "phind-codellama:latest", description: "Phind's fine-tuned CodeLlama.", params: "34B", reqRamGB: 22 },
  { name: "wizardcoder:latest", description: "WizardLM's advanced coding model.", params: "34B", reqRamGB: 22 },
  
  // Vision / Multimodal Models
  { name: "llava:latest", description: "Multimodal model that can understand and describe images.", params: "7B", reqRamGB: 7 },
  { name: "llava:13b", description: "Larger 13B version of LLaVA.", params: "13B", reqRamGB: 11 },
  { name: "bakllava:latest", description: "Mistral-based vision model.", params: "7B", reqRamGB: 7 },
  { name: "moondream:latest", description: "Tiny vision model that can run anywhere.", params: "1.8B", reqRamGB: 3 },
  
  // Embeddings
  { name: "nomic-embed-text:latest", description: "Excellent fast embedding model for RAG workflows.", params: "137M", reqRamGB: 2 },
  { name: "mxbai-embed-large:latest", description: "MixedBread's state-of-the-art embedding model.", params: "335M", reqRamGB: 2.5 },
  { name: "all-minilm:latest", description: "Classic lightweight embedding model.", params: "22M", reqRamGB: 1.5 },
  { name: "bge-m3:latest", description: "BAAI's massive multi-lingual embedding model.", params: "567M", reqRamGB: 3 },
  
  // Other Popular / Specialized
  { name: "dolphin-mixtral:8x7b", description: "Uncensored Mixtral model fine-tuned by Cognitive Computations.", params: "47B", reqRamGB: 28 },
  { name: "dolphin-llama3:8b", description: "Uncensored Llama 3.", params: "8B", reqRamGB: 7 },
  { name: "yi:34b", description: "01.AI's highly capable 34B model.", params: "34B", reqRamGB: 22 },
  { name: "command-r:latest", description: "Cohere's powerful 35B model tailored for RAG.", params: "35B", reqRamGB: 24 },
  { name: "command-r-plus:latest", description: "Cohere's massive 104B model.", params: "104B", reqRamGB: 60 },
  { name: "nous-hermes2:latest", description: "NousResearch's fine-tune of Mistral.", params: "7B", reqRamGB: 7 },
  { name: "openhermes:latest", description: "Teknium's excellent Mistral fine-tune.", params: "7B", reqRamGB: 7 },
  { name: "orca-mini:3b", description: "Microsoft's lightweight Orca research model.", params: "3B", reqRamGB: 4 },
  { name: "tinyllama:latest", description: "Incredibly small 1.1B model.", params: "1.1B", reqRamGB: 2 },
  { name: "qwen:7b", description: "First-gen Qwen 7B.", params: "7B", reqRamGB: 7 }
];
