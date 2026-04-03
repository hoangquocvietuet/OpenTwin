"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { getSettings, updateSettings, testConnection } from "@/lib/api";
import type { Settings } from "@/lib/types";

export default function SettingsPage() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [saving, setSaving] = useState(false);
  const [testResult, setTestResult] = useState<{ ok: boolean; latency_ms: number; error?: string } | null>(null);
  const [testing, setTesting] = useState(false);
  const [saveMsg, setSaveMsg] = useState("");

  useEffect(() => {
    getSettings().then(setSettings);
  }, []);

  if (!settings) return <div className="flex h-screen items-center justify-center text-zinc-400">Loading...</div>;

  async function handleSave() {
    if (!settings) return;
    setSaving(true);
    setSaveMsg("");
    try {
      await updateSettings(settings);
      setSaveMsg("Saved");
      setTimeout(() => setSaveMsg(""), 2000);
    } catch {
      setSaveMsg("Failed to save");
    }
    setSaving(false);
  }

  async function handleTest(baseUrl: string, apiKey: string) {
    setTesting(true);
    setTestResult(null);
    try {
      const result = await testConnection(baseUrl, apiKey);
      setTestResult(result);
    } catch {
      setTestResult({ ok: false, latency_ms: 0, error: "Request failed" });
    }
    setTesting(false);
  }

  function field(label: string, key: keyof Settings, type = "text") {
    return (
      <div>
        <label className="text-[13px] font-medium block mb-1.5">{label}</label>
        <Input
          type={type}
          value={settings![key]}
          onChange={(e) => setSettings({ ...settings!, [key]: e.target.value })}
          className="bg-zinc-950 border-zinc-800 focus-visible:border-accent-default text-sm"
        />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-zinc-950">
      <div className="max-w-xl mx-auto py-8 px-4">
        <div className="flex items-center gap-3 mb-6">
          <Link href="/" className="text-zinc-400 hover:text-zinc-200">
            <ArrowLeft className="h-4 w-4" />
          </Link>
          <h1 className="text-xl font-semibold">Settings</h1>
        </div>

        <Tabs defaultValue="chat-model">
          <TabsList className="bg-zinc-900 border border-zinc-800">
            <TabsTrigger value="chat-model">Chat Model</TabsTrigger>
            <TabsTrigger value="embedding">Embedding</TabsTrigger>
          </TabsList>

          <TabsContent value="chat-model" className="space-y-4 mt-4">
            {field("Base URL", "llm_base_url")}
            {field("Model", "llm_model")}
            {field("API Key", "llm_api_key", "password")}
            <div className="flex gap-2 items-center pt-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleTest(settings.llm_base_url, settings.llm_api_key)}
                disabled={testing}
                className="border-zinc-700"
              >
                {testing ? "Testing..." : "Test Connection"}
              </Button>
              <Button size="sm" onClick={handleSave} disabled={saving} className="bg-accent-default hover:bg-accent-hover">
                {saving ? "Saving..." : "Save"}
              </Button>
              {saveMsg && <span className="text-xs text-accent-default font-mono">{saveMsg}</span>}
            </div>
            {testResult && (
              <div className={`text-xs font-mono ${testResult.ok ? "text-accent-default" : "text-red-400"}`}>
                {testResult.ok ? `Connected (${testResult.latency_ms}ms)` : `Failed: ${testResult.error || "unreachable"}`}
              </div>
            )}
          </TabsContent>

          <TabsContent value="embedding" className="space-y-4 mt-4">
            {field("Embedding Model", "embedding_model")}
            {field("Embedding Base URL", "embedding_base_url")}
            {field("Embedding API Key", "embedding_api_key", "password")}
            <div className="flex gap-2 items-center pt-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleTest(settings.embedding_base_url, settings.embedding_api_key)}
                disabled={testing}
                className="border-zinc-700"
              >
                {testing ? "Testing..." : "Test Connection"}
              </Button>
              <Button size="sm" onClick={handleSave} disabled={saving} className="bg-accent-default hover:bg-accent-hover">
                {saving ? "Saving..." : "Save"}
              </Button>
            </div>
            {testResult && (
              <div className={`text-xs font-mono ${testResult.ok ? "text-accent-default" : "text-red-400"}`}>
                {testResult.ok ? `Connected (${testResult.latency_ms}ms)` : `Failed: ${testResult.error || "unreachable"}`}
              </div>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
