"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface ImportedSource {
  id: string;
  name: string;
  platform: string;
  enabled: boolean;
  created_at: string;
  total_messages: number;
  target_messages: number;
  train_chunks: number;
}

export default function ImportPage() {
  const [file, setFile] = useState<File | null>(null);
  const [sourceName, setSourceName] = useState("");
  const [targetName, setTargetName] = useState("");
  const [status, setStatus] = useState<{ ok: boolean; message: string } | null>(null);
  const [progress, setProgress] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);
  const [sources, setSources] = useState<ImportedSource[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  function refreshSources() {
    fetch("/api/v2/sources")
      .then((r) => (r.ok ? r.json() : []))
      .then(setSources)
      .catch(() => {});
  }

  useEffect(() => {
    refreshSources();
  }, []);

  async function handleImport() {
    if (!file) return;
    setImporting(true);
    setStatus(null);
    setProgress("Uploading...");

    const form = new FormData();
    form.append("file", file);
    form.append("source_name", sourceName);
    form.append("target_name", targetName);

    try {
      const res = await fetch("/api/v2/import", { method: "POST", body: form });

      if (!res.ok || !res.body) {
        const data = await res.json().catch(() => ({ detail: "Import failed" }));
        setStatus({ ok: false, message: data.detail || "Import failed" });
        setProgress(null);
        setImporting(false);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value, { stream: true });
        const lines = text.split("\n");

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const event = JSON.parse(line.slice(6));
            if (event.type === "progress") {
              setProgress(event.message);
            } else if (event.type === "done") {
              setStatus({ ok: true, message: `Imported! ${event.source_chunks} chunks from ${event.twin_name}` });
              setProgress(null);
              setFile(null);
              if (inputRef.current) inputRef.current.value = "";
              refreshSources();
            } else if (event.type === "error") {
              setStatus({ ok: false, message: event.message });
              setProgress(null);
            }
          } catch {
            // skip malformed lines
          }
        }
      }
    } catch {
      setStatus({ ok: false, message: "Connection error" });
      setProgress(null);
    }
    setImporting(false);
  }

  return (
    <div className="min-h-screen bg-zinc-950">
      <div className="max-w-xl mx-auto py-8 px-4">
        <div className="flex items-center gap-3 mb-6">
          <Link href="/" className="text-zinc-400 hover:text-zinc-200">
            <ArrowLeft className="h-4 w-4" />
          </Link>
          <h1 className="text-xl font-semibold">Import Data</h1>
        </div>

        <p className="text-zinc-400 text-sm mb-6">
          Upload a Facebook data export (.zip). Each import becomes a separate source.
        </p>

        <div className="space-y-4">
          <div>
            <label className="text-[13px] font-medium block mb-1.5">Upload .zip file</label>
            <input
              ref={inputRef}
              type="file"
              accept=".zip"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="text-sm text-zinc-400 file:mr-4 file:py-1.5 file:px-3 file:rounded-sm file:border file:border-zinc-700 file:text-[13px] file:bg-zinc-900 file:text-zinc-300 hover:file:bg-zinc-800"
            />
          </div>
          <div>
            <label className="text-[13px] font-medium block mb-1.5">Source name</label>
            <Input
              value={sourceName}
              onChange={(e) => setSourceName(e.target.value)}
              placeholder="e.g. 'Facebook main export'"
              className="bg-zinc-950 border-zinc-800 text-sm"
            />
          </div>
          <div>
            <label className="text-[13px] font-medium block mb-1.5">Target name (optional)</label>
            <Input
              value={targetName}
              onChange={(e) => setTargetName(e.target.value)}
              placeholder="Leave empty to auto-detect"
              className="bg-zinc-950 border-zinc-800 text-sm"
            />
          </div>

          <Button
            onClick={handleImport}
            disabled={!file || importing}
            className="bg-accent-default hover:bg-accent-hover"
          >
            {importing ? "Importing..." : "Import Source"}
          </Button>

          {progress && (
            <div className="text-sm text-zinc-400 font-mono flex items-center gap-2">
              <span className="inline-block w-3 h-3 border-2 border-accent-default border-t-transparent rounded-full animate-spin" />
              {progress}
            </div>
          )}

          {status && (
            <div className={`text-sm ${status.ok ? "text-accent-default" : "text-red-400"}`}>
              {status.message}
            </div>
          )}
        </div>

        {sources.length > 0 && (
          <div className="mt-10">
            <h2 className="text-sm font-medium text-zinc-400 uppercase tracking-wider mb-3">Imported Sources</h2>
            <div className="space-y-2">
              {sources.map((s) => (
                <div key={s.id} className="border border-zinc-800 rounded-md px-4 py-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium">{s.name}</div>
                      <div className="text-xs text-zinc-500 font-mono mt-0.5">
                        {s.platform} · {s.target_messages} messages · {s.train_chunks} chunks
                      </div>
                    </div>
                    <span className="text-[11px] text-zinc-600 font-mono">{s.id.slice(0, 8)}</span>
                  </div>
                  <div className="flex items-center gap-2 mt-2 pt-2 border-t border-zinc-800/50">
                    <button
                      onClick={async () => {
                        await fetch(`/api/v2/sources/${s.id}?enabled=${!s.enabled}`, { method: "PATCH" });
                        refreshSources();
                      }}
                      className={`text-[12px] font-mono px-2 py-0.5 rounded-sm transition-colors ${
                        s.enabled
                          ? "bg-accent-subtle text-accent-default hover:bg-accent-default/20"
                          : "bg-zinc-800 text-zinc-500 hover:bg-zinc-700"
                      }`}
                    >
                      {s.enabled ? "Enabled" : "Disabled"}
                    </button>
                    <button
                      onClick={async () => {
                        if (!confirm(`Delete source "${s.name}"? This removes all its chunks and embeddings.`)) return;
                        await fetch(`/api/v2/sources/${s.id}`, { method: "DELETE" });
                        refreshSources();
                      }}
                      className="text-[12px] font-mono px-2 py-0.5 rounded-sm text-zinc-500 hover:text-red-400 hover:bg-red-400/10 transition-colors"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
