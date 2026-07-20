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
} from "lucide-react";
import { authFetch } from "@/src/lib/auth";

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

export default function SettingsPage() {
  const [servers, setServers] = useState<MCPServer[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [expandedServer, setExpandedServer] = useState<string | null>(null);
  const [tools, setTools] = useState<Record<string, Tool[]>>({});
  const [loadingTools, setLoadingTools] = useState<string | null>(null);

  // Add server form
  const [showAddForm, setShowAddForm] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDescription, setNewDescription] = useState("");
  const [newUrl, setNewUrl] = useState("");
  const [addingServer, setAddingServer] = useState(false);
  const [feedback, setFeedback] = useState<{ type: "success" | "error"; message: string } | null>(null);

  const fetchServers = async () => {
    try {
      const res = await authFetch(`${BACKEND}/servers`);
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
  }, []);

  const refreshServers = async () => {
    setRefreshing(true);
    try {
      await authFetch(`${BACKEND}/servers/refresh`, { method: "POST" });
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
      });
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
      const res = await authFetch(`${BACKEND}/servers/${name}`, { method: "DELETE" });
      if (res.ok) {
        setServers((prev) => prev.filter((s) => s.name !== name));
        setFeedback({ type: "success", message: `Server '${name}' removed.` });
      }
    } catch {
      setFeedback({ type: "error", message: "Failed to remove server." });
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
      const res = await authFetch(`${BACKEND}/servers/${serverName}/tools`);
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
    <div className="h-full overflow-y-auto px-6 py-8">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-2xl font-semibold text-[var(--color2)]">Settings</h1>
            <p className="text-sm text-[var(--color3)] mt-1">
              Manage MCP servers that provide tools to Mosaic
            </p>
          </div>
          <button
            onClick={refreshServers}
            disabled={refreshing}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-[var(--input-bg)] border border-[var(--hover)] text-sm text-[var(--color1)] hover:bg-[var(--hover)] transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${refreshing ? "animate-spin" : ""}`} />
            Refresh All
          </button>
        </div>

        {/* Feedback toast */}
        {feedback && (
          <div
            className={`mb-4 px-4 py-3 rounded-xl text-sm ${
              feedback.type === "success"
                ? "bg-green-900/30 text-green-300 border border-green-800/50"
                : "bg-red-900/30 text-red-300 border border-red-800/50"
            }`}
          >
            {feedback.message}
          </div>
        )}

        {/* Server list */}
        <div className="space-y-3">
          {loading ? (
            <div className="text-[var(--color3)] text-sm py-8 text-center">Loading servers...</div>
          ) : servers.length === 0 ? (
            <div className="text-[var(--color3)] text-sm py-8 text-center">
              No MCP servers configured. Add one below.
            </div>
          ) : (
            servers.map((server) => (
              <div
                key={server.name}
                className="bg-[var(--input-bg)] border border-[var(--hover)] rounded-2xl overflow-hidden"
              >
                {/* Server header */}
                <div className="flex items-center gap-3 px-4 py-3">
                  <Server className="w-4 h-4 text-[var(--color3)] flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-[var(--color1)]">
                        {server.name}
                      </span>
                      {server.active ? (
                        <CheckCircle2 className="w-3.5 h-3.5 text-green-400" />
                      ) : (
                        <XCircle className="w-3.5 h-3.5 text-red-400" />
                      )}
                      {server.agent_loaded && (
                        <span className="text-xs bg-green-900/40 text-green-300 px-2 py-0.5 rounded-full">
                          agent active
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-[var(--color3)] truncate">{server.url}</p>
                    <p className="text-xs text-[var(--color3)] mt-0.5">{server.description}</p>
                  </div>
                  <div className="flex items-center gap-1">
                    {server.agent_loaded && (
                      <button
                        onClick={() => fetchTools(server.name)}
                        className="p-2 rounded-lg hover:bg-[var(--hover)] transition-colors text-[var(--color3)] hover:text-[var(--color1)]"
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
                      onClick={() => removeServer(server.name)}
                      className="p-2 rounded-lg hover:bg-[var(--hover)] transition-colors text-[var(--color3)] hover:text-red-400"
                      title="Remove server"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {/* Tools section */}
                {expandedServer === server.name && (
                  <div className="border-t border-[var(--hover)] px-4 py-3 bg-[var(--color4)]">
                    {loadingTools === server.name ? (
                      <p className="text-xs text-[var(--color3)]">Loading tools...</p>
                    ) : (tools[server.name] || []).length === 0 ? (
                      <p className="text-xs text-[var(--color3)]">No tools found.</p>
                    ) : (
                      <div className="space-y-2">
                        <p className="text-xs text-[var(--color3)] font-medium mb-2">
                          Available tools ({tools[server.name].length}):
                        </p>
                        {tools[server.name].map((tool) => (
                          <div
                            key={tool.name}
                            className="flex items-start gap-2 text-xs"
                          >
                            <Wrench className="w-3 h-3 text-[var(--color3)] mt-0.5 flex-shrink-0" />
                            <div>
                              <span className="text-[var(--color1)] font-medium">
                                {tool.name}
                              </span>
                              {tool.description && (
                                <p className="text-[var(--color3)] mt-0.5">
                                  {tool.description}
                                </p>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))
          )}
        </div>

        {/* Add server section */}
        <div className="mt-6">
          {!showAddForm ? (
            <button
              onClick={() => setShowAddForm(true)}
              className="flex items-center gap-2 px-4 py-3 w-full rounded-2xl border border-dashed border-[var(--color3)] text-sm text-[var(--color3)] hover:border-[var(--color2)] hover:text-[var(--color2)] transition-colors"
            >
              <Plus className="w-4 h-4" />
              Add MCP Server
            </button>
          ) : (
            <div className="bg-[var(--input-bg)] border border-[var(--hover)] rounded-2xl p-4 space-y-3">
              <h3 className="text-sm font-medium text-[var(--color1)]">Add MCP Server</h3>
              <div className="space-y-2">
                <input
                  type="text"
                  placeholder="Server name (e.g. my_server)"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  className="w-full px-3 py-2 rounded-xl bg-[var(--color4)] border border-[var(--hover)] text-sm text-[var(--color1)] placeholder-[var(--color3)] focus:outline-none focus:border-[var(--color2)]"
                />
                <input
                  type="text"
                  placeholder="URL (e.g. http://localhost:8000/sse)"
                  value={newUrl}
                  onChange={(e) => setNewUrl(e.target.value)}
                  className="w-full px-3 py-2 rounded-xl bg-[var(--color4)] border border-[var(--hover)] text-sm text-[var(--color1)] placeholder-[var(--color3)] focus:outline-none focus:border-[var(--color2)]"
                />
                <input
                  type="text"
                  placeholder="Description (optional)"
                  value={newDescription}
                  onChange={(e) => setNewDescription(e.target.value)}
                  className="w-full px-3 py-2 rounded-xl bg-[var(--color4)] border border-[var(--hover)] text-sm text-[var(--color1)] placeholder-[var(--color3)] focus:outline-none focus:border-[var(--color2)]"
                />
              </div>
              <div className="flex gap-2 pt-1">
                <button
                  onClick={addServer}
                  disabled={!newName.trim() || !newUrl.trim() || addingServer}
                  className="px-4 py-2 rounded-xl bg-[var(--color2)] text-[var(--color4)] text-sm font-medium disabled:opacity-40 hover:opacity-80 transition-opacity"
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
                  className="px-4 py-2 rounded-xl text-sm text-[var(--color3)] hover:text-[var(--color1)] transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
