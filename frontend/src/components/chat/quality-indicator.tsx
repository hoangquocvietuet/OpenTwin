"use client";

import { useState } from "react";
import type { RetrievalMetadata } from "@/lib/types";

interface QualityIndicatorProps {
  metadata: RetrievalMetadata;
}

export function QualityIndicator({ metadata }: QualityIndicatorProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="text-xs font-mono text-zinc-500">
      <button
        onClick={() => setExpanded(!expanded)}
        className="hover:text-accent-default transition-colors duration-100"
      >
        {expanded ? "▾" : "▸"} {metadata.chunks} chunks ({metadata.avg_similarity} avg)
        {metadata.pipeline && metadata.intent && (
          <span className="ml-1.5 text-zinc-400">· {metadata.intent}</span>
        )}
      </button>
      {expanded && (
        <div className="mt-1.5 space-y-1.5">
          {metadata.pipeline && (
            <div className="text-zinc-500">
              intent: {metadata.intent} | tone: {metadata.tone}
              {metadata.retries ? ` | ${metadata.retries} retries` : ""}
            </div>
          )}
          {metadata.retrieved && metadata.retrieved.length > 0 && (
            <div className="space-y-1">
              {metadata.retrieved.map((chunk, i) => (
                <div key={i} className="border-l-2 border-zinc-700 pl-2 py-0.5 text-zinc-500 text-[11px] leading-relaxed">
                  <span className="text-zinc-600">#{i + 1}</span>{" "}
                  <span className="text-zinc-400">{chunk.document}</span>
                  <span className="text-zinc-600 ml-1">({(1 - chunk.distance).toFixed(3)} sim)</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
