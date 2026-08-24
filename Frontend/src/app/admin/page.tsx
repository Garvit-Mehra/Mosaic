"use client";

import { useState, useEffect } from "react";
import {
  Shield,
  Activity,
  Trash2,
  AlertTriangle,
  Server,
  Settings,
  RefreshCw,
} from "lucide-react";
import { authFetch } from "@/src/lib/auth";
import { useSession } from "next-auth/react";
import { motion, AnimatePresence } from "framer-motion";

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

const butterySpring = {
  type: "spring",
  stiffness: 400,
  damping: 30,
  mass: 1,
};

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
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-[var(--color3)] animate-pulse">
        Initializing secure environment...
      </div>
    );
  }

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.1 } }
  };
  const itemVariants = {
    hidden: { opacity: 0, y: 15, scale: 0.98 },
    visible: { opacity: 1, y: 0, scale: 1, transition: butterySpring as any }
  };

  return (
    <div className="h-full overflow-y-auto px-6 py-12 bg-transparent">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="flex items-center gap-4 mb-10">
          <div className="w-12 h-12 rounded-full mosaic-panel flex items-center justify-center">
            <Shield className="w-6 h-6 text-[var(--color1)]" />
          </div>
          <div>
            <h1 className="text-3xl font-semibold text-[var(--color1)] tracking-tight">Admin Console</h1>
            <p className="text-sm text-[var(--color3)] mt-1">System diagnostics and runtime controls.</p>
          </div>
        </div>

        <AnimatePresence>
          {feedback && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={butterySpring as any}
              className="mb-6 px-4 py-3 rounded-2xl text-sm shadow-lg bg-emerald-900/40 text-emerald-300 border border-emerald-500/30 backdrop-blur-md"
            >
              {feedback}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Tabs */}
        <div className="flex gap-2 mb-8 mosaic-panel p-1.5 w-fit rounded-2xl">
          {(["overview", "logs", "errors", "config"] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-5 py-2 rounded-xl text-sm font-medium transition-all ${
                activeTab === tab
                  ? "bg-[rgba(255,255,255,0.1)] text-[var(--color1)] shadow-sm"
                  : "text-[var(--color3)] hover:text-[var(--color1)] hover:bg-[rgba(255,255,255,0.05)]"
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>

        <AnimatePresence mode="wait">
          {/* Overview Tab */}
          {activeTab === "overview" && status && (
            <motion.div 
              key="overview"
              variants={containerVariants}
              initial="hidden"
              animate="visible"
              exit="hidden"
              className="grid grid-cols-1 md:grid-cols-2 gap-5"
            >
              {/* System info */}
              <motion.div variants={itemVariants} className="mosaic-panel rounded-3xl p-6">
                <h3 className="text-sm font-semibold text-[var(--color1)] mb-4 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-blue-400" /> System Metrics
                </h3>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="flex flex-col"><span className="text-[var(--color3)] text-xs mb-1 uppercase tracking-wider font-semibold">Model</span> <span className="text-[var(--color1)]">{status.system.model}</span></div>
                  <div className="flex flex-col"><span className="text-[var(--color3)] text-xs mb-1 uppercase tracking-wider font-semibold">Python</span> <span className="text-[var(--color1)]">{status.system.python_version}</span></div>
                  <div className="flex flex-col"><span className="text-[var(--color3)] text-xs mb-1 uppercase tracking-wider font-semibold">Conversations</span> <span className="text-[var(--color1)] text-xl">{status.conversation_count}</span></div>
                  <div className="flex flex-col"><span className="text-[var(--color3)] text-xs mb-1 uppercase tracking-wider font-semibold">Platform</span> <span className="text-[var(--color1)]">{status.system.platform.split('-')[0]}</span></div>
                </div>
              </motion.div>

              {/* Agents */}
              <motion.div variants={itemVariants} className="mosaic-panel rounded-3xl p-6">
                <h3 className="text-sm font-semibold text-[var(--color1)] mb-4 flex items-center gap-2">
                  <Server className="w-4 h-4 text-purple-400" /> Active Agents
                </h3>
                <div className="flex flex-wrap gap-2">
                  {status.agents.map((a) => (
                    <span key={a} className="px-3 py-1.5 rounded-xl text-xs bg-emerald-500/10 text-emerald-300 border border-emerald-500/20 shadow-sm font-medium">
                      {a}
                    </span>
                  ))}
                </div>
                {status.inactive_servers.length > 0 && (
                  <div className="mt-5">
                    <p className="text-xs text-[var(--color3)] mb-2 font-semibold uppercase tracking-wider">Inactive MCPs:</p>
                    <div className="flex flex-wrap gap-2">
                      {status.inactive_servers.map((s) => (
                        <span key={s} className="px-3 py-1.5 rounded-xl text-xs bg-red-500/10 text-red-400 border border-red-500/20 shadow-sm font-medium">
                          {s}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </motion.div>

              {/* Danger zone */}
              <motion.div variants={itemVariants} className="mosaic-panel rounded-3xl p-6 md:col-span-2 border-red-500/30 bg-[rgba(255,0,0,0.02)]">
                <h3 className="text-sm font-semibold text-red-400 mb-3 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4" /> Danger Zone
                </h3>
                <p className="text-sm text-[var(--color3)] mb-4">
                  These actions are destructive and cannot be reversed. Use with extreme caution.
                </p>
                <button
                  onClick={clearAllConversations}
                  disabled={clearing}
                  className="flex items-center gap-2 px-5 py-2.5 rounded-2xl bg-red-900/40 text-red-300 text-sm font-semibold hover:bg-red-900/60 transition-colors disabled:opacity-50 border border-red-500/30 shadow-lg shadow-red-900/20"
                >
                  <Trash2 className="w-4 h-4" />
                  {clearing ? "Clearing Data..." : "Clear All Conversations"}
                </button>
              </motion.div>
            </motion.div>
          )}

          {/* Logs Tab */}
          {activeTab === "logs" && (
            <motion.div 
              key="logs"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={butterySpring as any}
              className="space-y-4"
            >
              <div className="flex justify-between items-center px-1">
                <h3 className="text-sm font-medium text-[var(--color1)]">Application Logs (last 80 lines)</h3>
                <button onClick={loadLogs} className="text-xs text-[var(--color3)] hover:text-[var(--color1)] flex items-center gap-1.5 mosaic-glass-button px-3 py-1.5">
                  <RefreshCw className="w-3 h-3" /> Refresh
                </button>
              </div>
              <div className="mosaic-panel rounded-3xl p-6 max-h-[60vh] overflow-y-auto font-mono text-xs leading-relaxed bg-[rgba(0,0,0,0.3)] border border-[rgba(255,255,255,0.05)] shadow-inner">
                {logs.length === 0 ? (
                  <p className="text-[var(--color3)]">No logs yet.</p>
                ) : (
                  logs.map((line, i) => (
                    <div key={i} className={`py-1 border-b border-[rgba(255,255,255,0.02)] last:border-0 ${line.includes("ERROR") ? "text-red-400" : line.includes("WARNING") ? "text-yellow-400" : "text-[var(--color3)]"}`}>
                      {line}
                    </div>
                  ))
                )}
              </div>
            </motion.div>
          )}

          {/* Errors Tab */}
          {activeTab === "errors" && (
            <motion.div 
              key="errors"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={butterySpring as any}
              className="space-y-4"
            >
              <div className="flex justify-between items-center px-1">
                <h3 className="text-sm font-medium text-[var(--color1)]">Error Logs</h3>
                <button onClick={loadErrorLogs} className="text-xs text-[var(--color3)] hover:text-[var(--color1)] flex items-center gap-1.5 mosaic-glass-button px-3 py-1.5">
                  <RefreshCw className="w-3 h-3" /> Refresh
                </button>
              </div>
              <div className="mosaic-panel rounded-3xl p-6 max-h-[60vh] overflow-y-auto font-mono text-xs leading-relaxed bg-[rgba(0,0,0,0.3)] border border-[rgba(255,255,255,0.05)] shadow-inner">
                {errorLogs.length === 0 ? (
                  <p className="text-emerald-400">No errors logged. All systems nominal. 🎉</p>
                ) : (
                  errorLogs.map((line, i) => (
                    <div key={i} className="py-1 border-b border-[rgba(255,255,255,0.02)] last:border-0 text-red-400">{line}</div>
                  ))
                )}
              </div>
            </motion.div>
          )}

          {/* Config Tab */}
          {activeTab === "config" && config && (
            <motion.div 
              key="config"
              initial={{ opacity: 0, y: 15, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              transition={butterySpring as any}
              className="mosaic-panel rounded-3xl p-6"
            >
              <h3 className="text-sm font-semibold text-[var(--color1)] mb-5 flex items-center gap-2">
                <Settings className="w-4 h-4 text-orange-400" /> Runtime Configuration
              </h3>
              <div className="grid grid-cols-1 gap-1 text-sm font-mono">
                {Object.entries(config).map(([key, value]) => (
                  <div key={key} className="flex flex-col sm:flex-row sm:items-center sm:justify-between py-3 border-b border-[var(--glass-border)] last:border-0 hover:bg-[rgba(255,255,255,0.02)] px-2 rounded-lg transition-colors">
                    <span className="text-[var(--color3)] mb-1 sm:mb-0 font-medium">{key}</span>
                    <span className={`font-semibold ${typeof value === 'boolean' ? (value ? 'text-emerald-400' : 'text-red-400') : 'text-[var(--color1)]'}`}>
                      {typeof value === "boolean" ? (value ? "Enabled" : "Disabled") : Array.isArray(value) ? value.join(", ") : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
