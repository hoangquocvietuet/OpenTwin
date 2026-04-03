"use client";

import { cn } from "@/lib/utils";
import { QualityIndicator } from "./quality-indicator";
import type { RetrievalMetadata } from "@/lib/types";

interface MessageProps {
  role: "user" | "assistant";
  content: string;
  twinName?: string;
  mode?: "answer" | "rewrite";
  metadata?: RetrievalMetadata | null;
}

export function Message({ role, content, twinName, mode, metadata }: MessageProps) {
  const isUser = role === "user";

  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div className={cn("max-w-[80%] space-y-1")}>
        {isUser && mode && (
          <div className="flex justify-end">
            <span className={cn(
              "text-[10px] font-mono px-1.5 py-0.5 rounded-sm",
              mode === "rewrite"
                ? "bg-amber-500/10 text-amber-400"
                : "bg-accent-subtle text-accent-default"
            )}>
              {mode}
            </span>
          </div>
        )}
        {!isUser && twinName && (
          <span className="text-[11px] font-medium text-zinc-400 uppercase tracking-wider">
            {twinName}
          </span>
        )}
        <div
          className={cn(
            "px-3.5 py-2.5 rounded-md text-sm leading-relaxed",
            isUser ? "bg-user-msg" : "bg-assistant-msg"
          )}
        >
          {content}
        </div>
        {!isUser && metadata && <QualityIndicator metadata={metadata} />}
      </div>
    </div>
  );
}
