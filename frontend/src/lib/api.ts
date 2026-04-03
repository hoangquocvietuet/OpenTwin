import type { Conversation, Message, Settings, TestConnectionResult } from "./types";

const API_BASE = "/api/v2";

export async function listConversations(): Promise<Conversation[]> {
  const res = await fetch(`${API_BASE}/conversations`);
  if (!res.ok) throw new Error("Failed to load conversations");
  return res.json();
}

export async function createConversation(title: string = "New Chat"): Promise<{ id: string; title: string }> {
  const res = await fetch(`${API_BASE}/conversations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  if (!res.ok) throw new Error("Failed to create conversation");
  return res.json();
}

export async function deleteConversation(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/conversations/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete conversation");
}

export async function getMessages(conversationId: string, limit = 50, beforeId?: number): Promise<Message[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (beforeId) params.set("before_id", String(beforeId));
  const res = await fetch(`${API_BASE}/conversations/${conversationId}/messages?${params}`);
  if (!res.ok) throw new Error("Failed to load messages");
  return res.json();
}

export async function getSettings(): Promise<Settings> {
  const res = await fetch(`${API_BASE}/settings`);
  if (!res.ok) throw new Error("Failed to load settings");
  return res.json();
}

export async function updateSettings(settings: Partial<Settings>): Promise<void> {
  const res = await fetch(`${API_BASE}/settings`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings),
  });
  if (!res.ok) throw new Error("Failed to save settings");
}

export async function testConnection(baseUrl: string, apiKey: string): Promise<TestConnectionResult> {
  const res = await fetch(`${API_BASE}/test-connection`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ base_url: baseUrl, api_key: apiKey }),
  });
  if (!res.ok) throw new Error("Failed to test connection");
  return res.json();
}
