"use client";

import { useState, useEffect } from "react";
import {
  Shield,
  Activity,
  Trash2,
  FileText,
  AlertTriangle,
  Server,
  Settings,
  RefreshCw,
} from "lucide-react";
import { authFetch } from "@/src/lib/auth";
import { useSession } from "next-auth/react";

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL;

interface SystemStatus {
  system: { python_version: string; platform: string; model: string };
  agents: string[];
  inactive_servers: string[];
  server_configs: { name: string; url: string }[];
  conversation_count: number;
}

interface Config {
  model: string;
  environment: string;
  token_expire_hours: number;
  login_rate_limit: number;
  login_rate_window_sec: number;
  allowed_origins: string[];
  log_level: string;
  tavily_key_set: boolean;
  jwt_secret_set: boolean;
  admin_user: string;
  normal_user: string;
}

export default function AdminPage() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [config, setConfig] = useState<Config | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [errorLogs, setErrorLogs] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<"overview" | "logs" | "errors" | "config">("overview");
  const [loading, setLoading] = useState(true);
  const [clearing, setClearing] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);
  const { data: session } = useSession();
  const token = (session as any)?.backendToken;

  useEffect(() => {
    // Admin check handled by middleware — just load data
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [statusRes, configRes] = await Promise.all([
        authFetch(`${BACKEND}/admin/status`, {}, token),
        authFetch(`${BACKEND}/admin/config`, {}, token),
      ]);
      if (statusRes.ok) setStatus(await statusRes.json());
      if (configRes.ok) setConfig(await configRes.json());
    } catch {
      // silent
    } finally {
      setLoading(false);
    }
  };

  const loadLogs = async () => {
    const res = await authFetch(`${BACKEND}/admin/logs?lines=80`, {}, token);
    if (res.ok) {
      const data = await res.json();
      setLogs(data.logs);
    }
  };

  const loadErrorLogs = async () => {
    const res = await authFetch(`${BACKEND}/admin/logs/errors?lines=50`, {}, token);
    if (res.ok) {
      const data = await res.json();
      setErrorLogs(data.logs);
    }
  };

  const clearAllConversations = async () => {
    if (!confirm("Delete ALL conversations for ALL users? This cannot be undone.")) return;
    setClearing(true);
    try {
      const res = await authFetch(`${BACKEND}/admin/conversations/clear`, { method: "DELETE" }, token);
      if (res.ok) {
        const data = await res.json();
        setFeedback(data.message);
        loadData();
      }
    } catch {
      setFeedback("Failed to clear conversations.");
    } finally {
      setClearing(false);
      setTimeout(() => setFeedback(null), 4000);
    }
  };

  useEffect(() => {
    if (activeTab === "logs") loadLogs();
    if (activeTab === "errors") loadErrorLogs();
  }, [activeTab]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-[var(--color3)]">
        Loading admin panel...
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto px-6 py-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="flex items-center gap-3 mb-6">
          <Shield className="w-6 h-6 text-[var(--color2)]" />
          <h1 className="text-2xl font-semibold text-[var(--color2)]">Admin Panel</h1>
        </div>

        {feedback && (
          <div className="mb-4 px-4 py-3 rounded-xl text-sm bg-green-900/30 text-green-300 border border-green-800/50">
            {feedback}
          </div>
        )}

        {/* Tabs */}
        <div className="flex gap-1 mb-6 bg-[var(--input-bg)] rounded-xl p-1 w-fit">
          {(["overview", "logs", "errors", "config"] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 rounded-lg text-sm transition-colors ${
                activeTab === tab
                  ? "bg-[var(--color2)] text-[var(--color4)]"
                  : "text-[var(--color3)] hover:text-[var(--color1)]"
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>

        {/* Overview Tab */}
        {activeTab === "overview" && status && (
          <div className="space-y-4">
            {/* System info */}
            <div className="bg-[var(--input-bg)] border border-[var(--hover)] rounded-2xl p-4">
              <h3 className="text-sm font-medium text-[var(--color1)] mb-3 flex items-center gap-2">
                <Activity className="w-4 h-4" /> System
              </h3>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div><span className="text-[var(--color3)]">Model:</span> <span className="text-[var(--color1)]">{status.system.model}</span></div>
                <div><span className="text-[var(--color3)]">Python:</span> <span className="text-[var(--color1)]">{status.system.python_version}</span></div>
                <div><span className="text-[var(--color3)]">Conversations:</span> <span className="text-[var(--color1)]">{status.conversation_count}</span></div>
                <div><span className="text-[var(--color3)]">Platform:</span> <span className="text-[var(--color1)]">{status.system.platform.split('-')[0]}</span></div>
              </div>
            </div>

            {/* Agents */}
            <div className="bg-[var(--input-bg)] border border-[var(--hover)] rounded-2xl p-4">
              <h3 className="text-sm font-medium text-[var(--color1)] mb-3 flex items-center gap-2">
                <Server className="w-4 h-4" /> Active Agents
              </h3>
              <div className="flex flex-wrap gap-2">
                {status.agents.map((a) => (
                  <span key={a} className="px-3 py-1 rounded-full text-xs bg-green-900/30 text-green-300 border border-green-800/50">
                    {a}
                  </span>
                ))}
              </div>
              {status.inactive_servers.length > 0 && (
                <div className="mt-3">
                  <p className="text-xs text-[var(--color3)] mb-1">Inactive:</p>
                  <div className="flex flex-wrap gap-2">
                    {status.inactive_servers.map((s) => (
                      <span key={s} className="px-3 py-1 rounded-full text-xs bg-red-900/30 text-red-300 border border-red-800/50">
                        {s}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Danger zone */}
            <div className="bg-red-950/20 border border-red-900/30 rounded-2xl p-4">
              <h3 className="text-sm font-medium text-red-300 mb-3 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4" /> Danger Zone
              </h3>
              <button
                onClick={clearAllConversations}
                disabled={clearing}
                className="flex items-center gap-2 px-4 py-2 rounded-xl bg-red-900/40 text-red-300 text-sm hover:bg-red-900/60 transition-colors disabled:opacity-50"
              >
                <Trash2 className="w-4 h-4" />
                {clearing ? "Clearing..." : "Clear All Conversations"}
              </button>
            </div>
          </div>
        )}

        {/* Logs Tab */}
        {activeTab === "logs" && (
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <h3 className="text-sm text-[var(--color3)]">Application Logs (last 80 lines)</h3>
              <button onClick={loadLogs} className="text-xs text-[var(--color3)] hover:text-[var(--color1)] flex items-center gap-1">
                <RefreshCw className="w-3 h-3" /> Refresh
              </button>
            </div>
            <div className="bg-[var(--color4)] border border-[var(--hover)] rounded-xl p-4 max-h-[60vh] overflow-y-auto font-mono text-xs">
              {logs.length === 0 ? (
                <p className="text-[var(--color3)]">No logs yet.</p>
              ) : (
                logs.map((line, i) => (
                  <div key={i} className={`py-0.5 ${line.includes("ERROR") ? "text-red-400" : line.includes("WARNING") ? "text-yellow-400" : "text-[var(--color3)]"}`}>
                    {line}
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {/* Errors Tab */}
        {activeTab === "errors" && (
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <h3 className="text-sm text-[var(--color3)]">Error Logs</h3>
              <button onClick={loadErrorLogs} className="text-xs text-[var(--color3)] hover:text-[var(--color1)] flex items-center gap-1">
                <RefreshCw className="w-3 h-3" /> Refresh
              </button>
            </div>
            <div className="bg-[var(--color4)] border border-[var(--hover)] rounded-xl p-4 max-h-[60vh] overflow-y-auto font-mono text-xs">
              {errorLogs.length === 0 ? (
                <p className="text-green-400">No errors logged. 🎉</p>
              ) : (
                errorLogs.map((line, i) => (
                  <div key={i} className="py-0.5 text-red-400">{line}</div>
                ))
              )}
            </div>
          </div>
        )}

        {/* Config Tab */}
        {activeTab === "config" && config && (
          <div className="bg-[var(--input-bg)] border border-[var(--hover)] rounded-2xl p-4">
            <h3 className="text-sm font-medium text-[var(--color1)] mb-3 flex items-center gap-2">
              <Settings className="w-4 h-4" /> Runtime Configuration
            </h3>
            <div className="grid grid-cols-1 gap-2 text-sm font-mono">
              {Object.entries(config).map(([key, value]) => (
                <div key={key} className="flex justify-between py-1 border-b border-[var(--hover)]">
                  <span className="text-[var(--color3)]">{key}</span>
                  <span className="text-[var(--color1)]">
                    {typeof value === "boolean" ? (value ? "✓" : "✗") : Array.isArray(value) ? value.join(", ") : String(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
