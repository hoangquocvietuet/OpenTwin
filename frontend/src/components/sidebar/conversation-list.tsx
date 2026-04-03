"use client";

import { cn } from "@/lib/utils";
import type { Conversation } from "@/lib/types";

interface ConversationListProps {
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

function groupByDate(
  conversations: Conversation[]
): Record<string, Conversation[]> {
  const groups: Record<string, Conversation[]> = {};
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today.getTime() - 86400000);
  const weekAgo = new Date(today.getTime() - 7 * 86400000);
  const monthAgo = new Date(today.getTime() - 30 * 86400000);

  for (const conv of conversations) {
    const date = conv.updated_at ? new Date(conv.updated_at) : new Date(0);
    let label: string;
    if (date >= today) label = "Today";
    else if (date >= yesterday) label = "Yesterday";
    else if (date >= weekAgo) label = "Last 7 days";
    else if (date >= monthAgo) label = "Last 30 days";
    else label = "Older";

    if (!groups[label]) groups[label] = [];
    groups[label].push(conv);
  }
  return groups;
}

export function ConversationList({
  conversations,
  activeId,
  onSelect,
  onDelete,
}: ConversationListProps) {
  if (conversations.length === 0) {
    return (
      <p className="px-3 py-4 text-zinc-500 text-xs">
        No conversations yet. Start chatting!
      </p>
    );
  }

  const groups = groupByDate(conversations);
  const order = ["Today", "Yesterday", "Last 7 days", "Last 30 days", "Older"];

  return (
    <div className="flex-1 overflow-y-auto">
      {order.map((label) =>
        groups[label] ? (
          <div key={label}>
            <div className="px-3 pt-3 pb-1 text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
              {label}
            </div>
            {groups[label].map((conv) => (
              <button
                key={conv.id}
                onClick={() => onSelect(conv.id)}
                className={cn(
                  "w-full text-left px-3 py-2 text-[13px] truncate hover:bg-zinc-800/50 transition-colors duration-100 group flex items-center justify-between",
                  activeId === conv.id && "bg-accent-subtle"
                )}
              >
                <span className="truncate">{conv.title}</span>
                <span
                  role="button"
                  tabIndex={0}
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(conv.id);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.stopPropagation();
                      onDelete(conv.id);
                    }
                  }}
                  className="opacity-0 group-hover:opacity-100 text-zinc-500 hover:text-red-400 text-xs ml-2 shrink-0"
                  aria-label="Delete conversation"
                >
                  ×
                </span>
              </button>
            ))}
          </div>
        ) : null
      )}
    </div>
  );
}
