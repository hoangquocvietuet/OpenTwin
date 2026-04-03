export interface Conversation {
  id: string;
  title: string;
  created_at: string | null;
  updated_at: string | null;
  last_message: string | null;
  last_message_at: string | null;
}

export interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
  retrieval_metadata: RetrievalMetadata | null;
  created_at: string | null;
}

export interface RetrievedChunk {
  document: string;
  distance: number;
}

export interface RetrievalMetadata {
  chunks: number;
  avg_similarity: number;
  retrieved?: RetrievedChunk[];
  pipeline?: boolean;
  intent?: string;
  tone?: string;
  retries?: number;
}

export interface Settings {
  llm_base_url: string;
  llm_model: string;
  llm_api_key: string;
  embedding_model: string;
  embedding_base_url: string;
  embedding_api_key: string;
}

export interface TestConnectionResult {
  ok: boolean;
  latency_ms: number;
  error?: string;
}
