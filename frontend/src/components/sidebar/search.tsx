"use client";

import { Search as SearchIcon, X } from "lucide-react";

interface SearchProps {
  value: string;
  onChange: (value: string) => void;
}

export function Search({ value, onChange }: SearchProps) {
  return (
    <div className="relative px-3 pb-2">
      <SearchIcon className="absolute left-5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-zinc-500" />
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Search..."
        className="w-full bg-zinc-950 border border-zinc-800 rounded-sm pl-8 pr-7 py-1.5 text-[13px] text-zinc-300 placeholder:text-zinc-600 focus:outline-none focus:border-accent-default"
      />
      {value && (
        <button
          onClick={() => onChange("")}
          className="absolute right-5 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      )}
    </div>
  );
}
