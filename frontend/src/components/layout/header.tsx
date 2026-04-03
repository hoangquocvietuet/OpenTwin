"use client";

import Link from "next/link";
import { cn } from "@/lib/utils";
import { Settings, Download, Menu } from "lucide-react";

interface HeaderProps {
  twinName: string;
  isConnected?: boolean | null;
  onMenuClick?: () => void;
}

export function Header({ twinName, isConnected, onMenuClick }: HeaderProps) {
  return (
    <header className="h-12 border-b border-zinc-800 flex items-center justify-between px-4 shrink-0">
      <div className="flex items-center gap-3">
        <button
          onClick={onMenuClick}
          className="lg:hidden text-zinc-400 hover:text-zinc-200 transition-colors"
          aria-label="Open menu"
        >
          <Menu className="h-5 w-5" />
        </button>
        <span className="font-semibold text-base">{twinName}</span>
      </div>
      <div className="flex items-center gap-3">
        <div
          className={cn(
            "w-2 h-2 rounded-full",
            isConnected === true && "bg-accent-default",
            isConnected === false && "bg-red-500",
            isConnected === null && "bg-amber-500"
          )}
          title={isConnected === true ? "Connected" : isConnected === false ? "Offline" : "Checking..."}
        />
        <Link href="/import" className="text-zinc-400 hover:text-zinc-200 transition-colors" aria-label="Import data">
          <Download className="h-4 w-4" />
        </Link>
        <Link href="/settings" className="text-zinc-400 hover:text-zinc-200 transition-colors" aria-label="Settings">
          <Settings className="h-4 w-4" />
        </Link>
      </div>
    </header>
  );
}
