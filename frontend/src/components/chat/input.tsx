"use client";

import { useState, useRef, type KeyboardEvent } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ArrowUp } from "lucide-react";

interface ChatInputProps {
  onSend: (message: string) => void;
  mode: "answer" | "rewrite";
  onModeToggle: () => void;
  disabled?: boolean;
}

export function ChatInput({
  onSend,
  mode,
  onModeToggle,
  disabled,
}: ChatInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  function handleSubmit() {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
    textareaRef.current?.focus();
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  return (
    <div className="border-t border-zinc-800 px-4 py-3">
      <div className="flex items-center gap-2 mb-2">
        <button
          onClick={onModeToggle}
          className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-accent-subtle text-accent-default hover:bg-accent-default/20 transition-colors duration-100"
        >
          {mode === "answer" ? "Answer" : "Rewrite"}
        </button>
      </div>
      <div className="flex gap-2 items-end">
        <Textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message..."
          className="min-h-[40px] max-h-[120px] resize-none bg-surface border-zinc-800 focus-visible:border-accent-default text-sm"
          rows={1}
          disabled={disabled}
        />
        <Button
          onClick={handleSubmit}
          disabled={!value.trim() || disabled}
          size="icon"
          className="h-10 w-10 bg-accent-default hover:bg-accent-hover shrink-0"
        >
          <ArrowUp className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
