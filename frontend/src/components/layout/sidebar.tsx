"use client";

import { useState } from "react";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ConversationList } from "@/components/sidebar/conversation-list";
import { Search } from "@/components/sidebar/search";
import type { Conversation } from "@/lib/types";

interface SidebarProps {
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onNewChat: () => void;
  isOpen?: boolean;
  onClose?: () => void;
}

export function Sidebar({
  conversations,
  activeId,
  onSelect,
  onDelete,
  onNewChat,
  isOpen,
  onClose,
}: SidebarProps) {
  const [search, setSearch] = useState("");

  const filtered = search
    ? conversations.filter((c) =>
        c.title.toLowerCase().includes(search.toLowerCase())
      )
    : conversations;

  const handleSelect = (id: string) => {
    onSelect(id);
    onClose?.();
  };

  const sidebarContent = (
    <div className="flex flex-col h-full">
      <div className="p-3 flex items-center gap-2">
        <Button
          onClick={onNewChat}
          className="flex-1 bg-accent-default hover:bg-accent-hover text-white text-[13px] h-9"
        >
          + New Chat
        </Button>
        <button
          onClick={onClose}
          className="lg:hidden p-1 text-zinc-400 hover:text-zinc-200 transition-colors"
          aria-label="Close sidebar"
        >
          <X className="h-5 w-5" />
        </button>
      </div>
      <Search value={search} onChange={setSearch} />
      <ConversationList
        conversations={filtered}
        activeId={activeId}
        onSelect={handleSelect}
        onDelete={onDelete}
      />
    </div>
  );

  return (
    <>
      {/* Desktop: inline sidebar (always visible on lg+) */}
      <aside className="hidden lg:flex flex-col w-[280px] border-r border-zinc-800 bg-surface shrink-0 h-full">
        {sidebarContent}
      </aside>

      {/* Mobile: overlay drawer */}
      {isOpen && (
        <div className="lg:hidden fixed inset-0 z-50 flex">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/60"
            onClick={onClose}
            aria-hidden="true"
          />
          {/* Drawer panel */}
          <div className="relative w-[280px] bg-surface border-r border-zinc-800 flex flex-col h-full">
            {sidebarContent}
          </div>
        </div>
      )}
    </>
  );
}
