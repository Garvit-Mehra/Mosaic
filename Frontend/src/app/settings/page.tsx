"use client";

import { useState, useEffect } from "react";
import {
  Server,
  Plus,
  Trash2,
  RefreshCw,
  CheckCircle2,
  XCircle,
  ChevronDown,
  ChevronRight,
  Wrench,
  Pencil,
  Check,
  X,
} from "lucide-react";
import { authFetch } from "@/src/lib/auth";
import { useSession } from "next-auth/react";
import { motion, AnimatePresence } from "framer-motion";

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL;

interface MCPServer {
  name: string;
  description: string;
  url: string;
  active: boolean;
  agent_loaded: boolean;
}

interface Tool {
  name: string;
  description: string;
}

const butterySpring = {
  type: "spring",
  stiffness: 400,
  damping: 30,
  mass: 1,
};

export default function SettingsPage() {
  const [servers, setServers] = useState<MCPServer[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [expandedServer, setExpandedServer] = useState<string | null>(null);
  const [tools, setTools] = useState<Record<string, Tool[]>>({});
  const [loadingTools, setLoadingTools] = useState<string | null>(null);
  const { data: session } = useSession();
  const token = (session as any)?.backendToken;

  // Add server form
  const [showAddForm, setShowAddForm] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDescription, setNewDescription] = useState("");
  const [newUrl, setNewUrl] = useState("");
  const [addingServer, setAddingServer] = useState(false);
  const [feedback, setFeedback] = useState<{ type: "success" | "error"; message: string } | null>(null);

  // Edit server
  const [editingServer, setEditingServer] = useState<string | null>(null);
  const [editUrl, setEditUrl] = useState("");
  const [editDescription, setEditDescription] = useState("");

  const fetchServers = async () => {
    try {
      const res = await authFetch(`${BACKEND}/servers`, {}, token);
      if (res.ok) {
        const data = await res.json();
        setServers(data.servers);
      }
    } catch {
      // silent
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchServers();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const refreshServers = async () => {
    setRefreshing(true);
    try {
      await authFetch(`${BACKEND}/servers/refresh`, { method: "POST" }, token);
      await fetchServers();
      setFeedback({ type: "success", message: "Servers refreshed." });
    } catch {
      setFeedback({ type: "error", message: "Failed to refresh servers." });
    } finally {
      setRefreshing(false);
      setTimeout(() => setFeedback(null), 3000);
    }
  };

  const addServer = async () => {
    if (!newName.trim() || !newUrl.trim()) return;
    setAddingServer(true);
    try {
      const res = await authFetch(`${BACKEND}/servers`, {
        method: "POST",
        body: JSON.stringify({
          name: newName.trim().toLowerCase().replace(/\s+/g, "_"),
          description: newDescription.trim() || `MCP server at ${newUrl.trim()}`,
          url: newUrl.trim(),
        }),
      }, token);
      const data = await res.json();
      if (res.ok) {
        setFeedback({ type: "success", message: data.message });
        setNewName("");
        setNewDescription("");
        setNewUrl("");
        setShowAddForm(false);
        await fetchServers();
      } else {
        setFeedback({ type: "error", message: data.detail || "Failed to add server." });
      }
    } catch {
      setFeedback({ type: "error", message: "Could not reach the backend." });
    } finally {
      setAddingServer(false);
      setTimeout(() => setFeedback(null), 4000);
    }
  };

  const removeServer = async (name: string) => {
    try {
      const res = await authFetch(`${BACKEND}/servers/${name}`, { method: "DELETE" }, token);
      if (res.ok) {
        setServers((prev) => prev.filter((s) => s.name !== name));
        setFeedback({ type: "success", message: `Server '${name}' removed.` });
      }
    } catch {
      setFeedback({ type: "error", message: "Failed to remove server." });
    }
    setTimeout(() => setFeedback(null), 3000);
  };

  const startEdit = (server: MCPServer) => {
    setEditingServer(server.name);
    setEditUrl(server.url);
    setEditDescription(server.description);
  };

  const cancelEdit = () => {
    setEditingServer(null);
    setEditUrl("");
    setEditDescription("");
  };

  const saveEdit = async (name: string) => {
    try {
      const res = await authFetch(`${BACKEND}/servers/${name}`, {
        method: "PATCH",
        body: JSON.stringify({ url: editUrl, description: editDescription }),
      }, token);
      if (res.ok) {
        setFeedback({ type: "success", message: `Server '${name}' updated.` });
        setEditingServer(null);
        await fetchServers();
      } else {
        const data = await res.json();
        setFeedback({ type: "error", message: data.detail || "Failed to update." });
      }
    } catch {
      setFeedback({ type: "error", message: "Could not reach backend." });
    }
    setTimeout(() => setFeedback(null), 3000);
  };

  const fetchTools = async (serverName: string) => {
    if (tools[serverName]) {
      setExpandedServer(expandedServer === serverName ? null : serverName);
      return;
    }
    setLoadingTools(serverName);
    setExpandedServer(serverName);
    try {
      const res = await authFetch(`${BACKEND}/servers/${serverName}/tools`, {}, token);
      if (res.ok) {
        const data = await res.json();
        setTools((prev) => ({ ...prev, [serverName]: data.tools }));
      } else {
        setTools((prev) => ({ ...prev, [serverName]: [] }));
      }
    } catch {
      setTools((prev) => ({ ...prev, [serverName]: [] }));
    } finally {
      setLoadingTools(null);
    }
  };

  return (
    <div className="h-full overflow-y-auto px-6 py-12 bg-transparent">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-semibold text-[var(--color1)] tracking-tight">MCP Servers</h1>
            <p className="text-sm text-[var(--color3)] mt-2">
              Manage external tools and agents connected to Mosaic.
            </p>
          </div>
          <button
            onClick={refreshServers}
            disabled={refreshing}
            className="flex items-center gap-2 px-5 py-2.5 mosaic-glass-button text-[var(--color1)] hover:bg-[var(--hover)] transition-colors disabled:opacity-50 text-sm font-medium"
          >
            <RefreshCw className={`w-4 h-4 ${refreshing ? "animate-spin" : ""}`} />
            Refresh All
          </button>
        </div>

        {/* Feedback toast */}
        <AnimatePresence>
          {feedback && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={butterySpring as any}
              className={`mb-6 px-4 py-3 rounded-2xl text-sm shadow-lg ${
                feedback.type === "success"
                  ? "bg-emerald-900/40 text-emerald-300 border border-emerald-500/30 backdrop-blur-md"
                  : "bg-red-900/40 text-red-300 border border-red-500/30 backdrop-blur-md"
              }`}
            >
              {feedback.message}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Server list */}
        <div className="space-y-4">
          {loading ? (
            <div className="text-[var(--color3)] text-sm py-12 text-center animate-pulse">Loading servers...</div>
          ) : servers.length === 0 ? (
            <div className="text-[var(--color3)] text-sm py-12 text-center mosaic-panel rounded-3xl">
              No MCP servers configured. Add one below.
            </div>
          ) : (
            <motion.div
              initial="hidden"
              animate="visible"
              variants={{
                visible: { transition: { staggerChildren: 0.05 } },
                hidden: {}
              }}
              className="space-y-4"
            >
              {servers.map((server) => (
                <motion.div
                  variants={{
                    hidden: { opacity: 0, y: 15, scale: 0.98 },
                    visible: { opacity: 1, y: 0, scale: 1, transition: butterySpring as any }
                  }}
                  key={server.name}
                  className="mosaic-panel rounded-3xl overflow-hidden transition-all hover:border-[var(--color3)]"
                >
                  {/* Server header */}
                  <div className="flex items-center gap-4 px-6 py-5">
                    <div className="w-10 h-10 rounded-full bg-[var(--hover)] border border-[var(--glass-border)] flex items-center justify-center flex-shrink-0">
                      <Server className="w-5 h-5 text-[var(--color1)]" />
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      {editingServer === server.name ? (
                        /* Edit mode */
                        <div className="space-y-3 mt-1">
                          <span className="text-sm font-semibold text-[var(--color1)]">{server.name}</span>
                          <input
                            type="text"
                            value={editUrl}
                            onChange={(e) => setEditUrl(e.target.value)}
                            placeholder="URL"
                            className="w-full px-4 py-2 rounded-xl bg-[var(--input-bg)] border border-[var(--glass-border)] text-sm text-[var(--color1)] mosaic-input-focus transition-all"
                          />
                          <input
                            type="text"
                            value={editDescription}
                            onChange={(e) => setEditDescription(e.target.value)}
                            placeholder="Description"
                            className="w-full px-4 py-2 rounded-xl bg-[var(--input-bg)] border border-[var(--glass-border)] text-sm text-[var(--color1)] mosaic-input-focus transition-all"
                          />
                        </div>
                      ) : (
                        /* View mode */
                        <>
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-base font-semibold text-[var(--color1)]">
                              {server.name}
                            </span>
                            {server.active ? (
                              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                            ) : (
                              <XCircle className="w-4 h-4 text-red-400" />
                            )}
                            {server.agent_loaded && (
                              <span className="text-xs bg-emerald-500/10 text-emerald-400 px-2 py-0.5 rounded-md border border-emerald-500/20 shadow-sm ml-2">
                                agent active
                              </span>
                            )}
                          </div>
                          <p className="text-sm text-[var(--color3)] truncate">{server.url}</p>
                          <p className="text-sm text-[var(--color3)] mt-1">{server.description}</p>
                        </>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      {editingServer === server.name ? (
                        <>
                          <button
                            onClick={() => saveEdit(server.name)}
                            className="p-2.5 rounded-xl hover:bg-[var(--hover)] transition-colors text-emerald-400 border border-transparent hover:border-[var(--glass-border)]"
                            title="Save"
                          >
                            <Check className="w-4 h-4" />
                          </button>
                          <button
                            onClick={cancelEdit}
                            className="p-2.5 rounded-xl hover:bg-[var(--hover)] transition-colors text-red-400 border border-transparent hover:border-[var(--glass-border)]"
                            title="Cancel"
                          >
                            <X className="w-4 h-4" />
                          </button>
                        </>
                      ) : (
                        <>
                          {server.agent_loaded && (
                            <button
                              onClick={() => fetchTools(server.name)}
                              className="p-2.5 rounded-xl hover:bg-[var(--hover)] transition-colors text-[var(--color3)] hover:text-[var(--color1)] border border-transparent hover:border-[var(--glass-border)]"
                              title="View tools"
                            >
                              {expandedServer === server.name ? (
                                <ChevronDown className="w-4 h-4" />
                              ) : (
                                <ChevronRight className="w-4 h-4" />
                              )}
                            </button>
                          )}
                          <button
                            onClick={() => startEdit(server)}
                            className="p-2.5 rounded-xl hover:bg-[var(--hover)] transition-colors text-[var(--color3)] hover:text-[var(--color1)] border border-transparent hover:border-[var(--glass-border)]"
                            title="Edit server"
                          >
                            <Pencil className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => removeServer(server.name)}
                            className="p-2.5 rounded-xl hover:bg-red-500/10 transition-colors text-[var(--color3)] hover:text-red-400 border border-transparent hover:border-red-500/20"
                            title="Remove server"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </>
                      )}
                    </div>
                  </div>

                  {/* Tools section */}
                  <AnimatePresence>
                    {expandedServer === server.name && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={butterySpring as any}
                        className="border-t border-[var(--glass-border)] bg-[rgba(255,255,255,0.02)] overflow-hidden"
                      >
                        <div className="px-6 py-4">
                          {loadingTools === server.name ? (
                            <p className="text-sm text-[var(--color3)] py-2">Loading tools...</p>
                          ) : (tools[server.name] || []).length === 0 ? (
                            <p className="text-sm text-[var(--color3)] py-2">No tools detected.</p>
                          ) : (
                            <div className="space-y-3">
                              <p className="text-sm text-[var(--color3)] font-medium">
                                {tools[server.name].length} tools available
                              </p>
                              <div className="max-h-64 overflow-y-auto space-y-2 pr-2">
                                {tools[server.name].map((tool) => (
                                  <div
                                    key={tool.name}
                                    className="flex items-center gap-3 text-sm px-4 py-2.5 rounded-xl bg-[var(--input-bg)] border border-[var(--glass-border)] hover:bg-[var(--hover)] transition-colors"
                                  >
                                    <Wrench className="w-4 h-4 text-[var(--color3)] flex-shrink-0" />
                                    <span className="text-[var(--color1)] font-medium">{tool.name}</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              ))}
            </motion.div>
          )}
        </div>

        {/* Add server section */}
        <div className="mt-8">
          <AnimatePresence mode="wait">
            {!showAddForm ? (
              <motion.button
                key="add-btn"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={butterySpring as any}
                onClick={() => setShowAddForm(true)}
                className="flex items-center justify-center gap-2 px-4 py-4 w-full rounded-3xl border border-dashed border-[var(--color3)] text-sm text-[var(--color3)] hover:border-[var(--color2)] hover:text-[var(--color1)] hover:bg-[var(--hover)] transition-all bg-[rgba(255,255,255,0.01)]"
              >
                <Plus className="w-5 h-5" />
                <span className="font-medium">Add New MCP Server</span>
              </motion.button>
            ) : (
              <motion.div 
                key="add-form"
                initial={{ opacity: 0, scale: 0.98, y: 10 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.98, y: -10 }}
                transition={butterySpring as any}
                className="mosaic-panel rounded-3xl p-6 space-y-5"
              >
                <h3 className="text-lg font-semibold text-[var(--color1)]">Add MCP Server</h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-xs font-medium text-[var(--color3)] mb-1.5 px-1">Server Name</label>
                    <input
                      type="text"
                      placeholder="e.g. my_server"
                      value={newName}
                      onChange={(e) => setNewName(e.target.value)}
                      className="w-full px-4 py-3 rounded-2xl bg-[var(--input-bg)] border border-[var(--glass-border)] text-sm text-[var(--color1)] placeholder-[var(--color3)] mosaic-input-focus transition-all"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-[var(--color3)] mb-1.5 px-1">Server URL</label>
                    <input
                      type="text"
                      placeholder="e.g. http://localhost:8000/sse"
                      value={newUrl}
                      onChange={(e) => setNewUrl(e.target.value)}
                      className="w-full px-4 py-3 rounded-2xl bg-[var(--input-bg)] border border-[var(--glass-border)] text-sm text-[var(--color1)] placeholder-[var(--color3)] mosaic-input-focus transition-all"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-[var(--color3)] mb-1.5 px-1">Description (Optional)</label>
                    <input
                      type="text"
                      placeholder="What tools does this provide?"
                      value={newDescription}
                      onChange={(e) => setNewDescription(e.target.value)}
                      className="w-full px-4 py-3 rounded-2xl bg-[var(--input-bg)] border border-[var(--glass-border)] text-sm text-[var(--color1)] placeholder-[var(--color3)] mosaic-input-focus transition-all"
                    />
                  </div>
                </div>
                <div className="flex gap-3 pt-2">
                  <button
                    onClick={addServer}
                    disabled={!newName.trim() || !newUrl.trim() || addingServer}
                    className="px-6 py-2.5 rounded-xl bg-blue-600 text-white text-sm font-semibold disabled:opacity-50 hover:bg-blue-500 shadow-lg shadow-blue-900/20 border border-blue-500/50 transition-all"
                  >
                    {addingServer ? "Adding..." : "Add Server"}
                  </button>
                  <button
                    onClick={() => {
                      setShowAddForm(false);
                      setNewName("");
                      setNewUrl("");
                      setNewDescription("");
                    }}
                    className="px-6 py-2.5 rounded-xl text-sm font-medium text-[var(--color3)] hover:text-[var(--color1)] hover:bg-[var(--hover)] transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
