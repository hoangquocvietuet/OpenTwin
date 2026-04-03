"use client";

import { useEffect, useRef } from "react";
import { Message } from "./message";
import type { RetrievalMetadata } from "@/lib/types";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  mode?: "answer" | "rewrite";
  metadata?: RetrievalMetadata | null;
}

interface MessageListProps {
  messages: ChatMessage[];
  twinName: string;
  isLoading?: boolean;
}

export function MessageList({
  messages,
  twinName,
  isLoading,
}: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <p className="text-zinc-500 text-sm">Ask your twin anything...</p>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
      {messages.map((msg, i) => (
        <Message
          key={i}
          role={msg.role}
          content={msg.content}
          twinName={twinName}
          mode={msg.mode}
          metadata={msg.metadata}
        />
      ))}
      {isLoading && (
        <div className="flex justify-start">
          <div className="bg-assistant-msg px-3.5 py-2.5 rounded-md">
            <span className="text-zinc-400 text-sm animate-pulse">
              Thinking...
            </span>
          </div>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}
