"use client";

import { useState, useCallback, useEffect } from "react";
import { MessageList } from "@/components/chat/message-list";
import { ChatInput } from "@/components/chat/input";
import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
  createConversation,
  listConversations,
  getMessages,
  deleteConversation,
} from "@/lib/api";
import type { Conversation, RetrievalMetadata } from "@/lib/types";

function WelcomeCard() {
  return (
    <div className="flex-1 flex items-center justify-center">
      <div className="text-center space-y-4 max-w-sm">
        <h2 className="text-xl font-semibold">OpenTwin</h2>
        <p className="text-zinc-400 text-sm">
          Import your chat data to create your digital twin. It responds in your voice.
        </p>
        <p className="text-xs font-mono text-zinc-500">
          <span className="text-accent-default">●</span> 100% local. Your data never leaves your machine.
        </p>
        <Link href="/import">
          <Button className="bg-accent-default hover:bg-accent-hover mt-2">Import Data →</Button>
        </Link>
      </div>
    </div>
  );
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  mode?: "answer" | "rewrite";
  metadata?: RetrievalMetadata | null;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [mode, setMode] = useState<"answer" | "rewrite">("answer");
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const refreshConversations = useCallback(() => {
    listConversations().then(setConversations).catch(() => {});
  }, []);

  useEffect(() => {
    refreshConversations();
  }, [refreshConversations]);

  const handleSelectConversation = useCallback((id: string) => {
    setConversationId(id);
    getMessages(id)
      .then((msgs) =>
        setMessages(
          msgs.map((m) => ({
            role: m.role,
            content: m.content,
            metadata: m.retrieval_metadata,
          }))
        )
      )
      .catch(() => {});
  }, []);

  const handleDeleteConversation = useCallback(
    (id: string) => {
      deleteConversation(id)
        .then(() => {
          refreshConversations();
          if (conversationId === id) {
            setConversationId(null);
            setMessages([]);
          }
        })
        .catch(() => {});
    },
    [conversationId, refreshConversations]
  );

  const handleNewChat = useCallback(() => {
    setConversationId(null);
    setMessages([]);
  }, []);

  const handleSend = useCallback(
    async (content: string) => {
      // Create conversation on first message
      let convId = conversationId;
      if (!convId) {
        const conv = await createConversation(content.slice(0, 50));
        convId = conv.id;
        setConversationId(convId);
      }

      setMessages((prev) => [...prev, { role: "user", content, mode }]);
      setIsLoading(true);

      try {
        const res = await fetch("/api/v2/chat/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ content, mode, conversation_id: convId }),
        });

        if (!res.ok || !res.body) {
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: "Failed to get response. Check Settings.",
            },
          ]);
          setIsLoading(false);
          return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let assistantText = "";
        let metadata: RetrievalMetadata | null = null;

        // Add empty assistant message
        setMessages((prev) => [...prev, { role: "assistant", content: "" }]);
        setIsLoading(false);

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const text = decoder.decode(value, { stream: true });
          const lines = text.split("\n");

          for (const line of lines) {
            if (line.startsWith("0:")) {
              // Text chunk
              const chunk = JSON.parse(line.slice(2));
              assistantText += chunk;
              setMessages((prev) => {
                const updated = [...prev];
                updated[updated.length - 1] = {
                  role: "assistant",
                  content: assistantText,
                };
                return updated;
              });
            } else if (line.startsWith("data: ")) {
              // Metadata or error
              const parsed = JSON.parse(line.slice(6));
              if (parsed.error) {
                setMessages((prev) => {
                  const updated = [...prev];
                  updated[updated.length - 1] = {
                    role: "assistant",
                    content: parsed.content,
                  };
                  return updated;
                });
              } else {
                metadata = parsed;
                setMessages((prev) => {
                  const updated = [...prev];
                  updated[updated.length - 1] = {
                    ...updated[updated.length - 1],
                    metadata,
                  };
                  return updated;
                });
              }
            }
          }
        }
        refreshConversations();
      } catch {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: "Connection error. Is the backend running?",
          },
        ]);
        setIsLoading(false);
      }
    },
    [conversationId, mode, refreshConversations]
  );

  return (
    <div className="flex h-screen">
      <Sidebar
        conversations={conversations}
        activeId={conversationId}
        onSelect={handleSelectConversation}
        onDelete={handleDeleteConversation}
        onNewChat={handleNewChat}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />
      <main className="flex-1 flex flex-col min-w-0">
        <Header
          twinName="OpenTwin"
          onMenuClick={() => setSidebarOpen(true)}
        />
        {messages.length === 0 && !conversationId ? (
          <WelcomeCard />
        ) : (
          <MessageList
            messages={messages}
            twinName="Twin"
            isLoading={isLoading}
          />
        )}
        <ChatInput
          onSend={handleSend}
          mode={mode}
          onModeToggle={() =>
            setMode((m) => (m === "answer" ? "rewrite" : "answer"))
          }
          disabled={isLoading}
        />
      </main>
    </div>
  );
}
