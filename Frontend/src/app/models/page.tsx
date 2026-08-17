"use client";

import React, { useState, useEffect } from "react";
import { Search, Server, Cpu, CheckCircle2, Download, ChevronRight, Hash, Database, Loader2, Plus, X } from "lucide-react";
import { useRouter } from "next/navigation";
import { CURATED_MODELS, CuratedModel } from "./data";

interface ModelDetails {
  parent_model: string;
  format: string;
  family: string;
  families: string[];
  parameter_size: string;
  quantization_level: string;
}

interface OllamaModel {
  name: string;
  model: string;
  modified_at: string;
  size: number;
  digest: string;
  details: ModelDetails;
}

const ITEMS_PER_PAGE = 15;

export default function ModelsPage() {
  const [installedModels, setInstalledModels] = useState<OllamaModel[]>([]);
  const [customModels, setCustomModels] = useState<CuratedModel[]>([]);
  const [sysRamBytes, setSysRamBytes] = useState<number>(0);
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState("name");
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState<string | null>(null);
  
  // Pagination
  const [currentPage, setCurrentPage] = useState(1);
  
  // Custom Modal
  const [showModal, setShowModal] = useState(false);
  const [customModelName, setCustomModelName] = useState("");
  
  const router = useRouter();
  const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8080";

  useEffect(() => {
    // Load custom models from localStorage
    const saved = localStorage.getItem("mosaic_custom_models");
    if (saved) {
      try {
        setCustomModels(JSON.parse(saved));
      } catch (e) {
        console.error("Failed to parse custom models", e);
      }
    }
  }, []);

  const fetchModels = async () => {
    try {
      const res = await fetch(`${BACKEND}/api/models/details`);
      if (res.ok) {
        const data = await res.json();
        setInstalledModels(data.models || []);
        if (data.system?.ram_bytes) setSysRamBytes(data.system.ram_bytes);
      }
    } catch (err) {
      console.error("Failed to fetch models", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleDownload = async (modelName: string) => {
    setDownloading(modelName);
    try {
      const res = await fetch(`${BACKEND}/api/models/pull`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: modelName })
      });
      if (res.ok) {
        await fetchModels(); // Refresh installed list
      } else {
        alert("Failed to download model.");
      }
    } catch (err) {
      console.error(err);
      alert("Failed to connect to backend to pull model.");
    } finally {
      setDownloading(null);
    }
  };

  const handleAddCustomModel = () => {
    if (!customModelName.trim()) return;
    const cleanName = customModelName.trim().toLowerCase();
    
    // Create custom entry
    const newCustom: CuratedModel = {
      name: cleanName,
      description: "Custom user-added model.",
      params: "Unknown",
      reqRamGB: 0
    };
    
    const updated = [...customModels, newCustom];
    setCustomModels(updated);
    localStorage.setItem("mosaic_custom_models", JSON.stringify(updated));
    
    setShowModal(false);
    setCustomModelName("");
    
    // Instantly try to download it
    handleDownload(cleanName);
  };

  const handleUseModel = (modelName: string) => {
    localStorage.setItem("mosaic_selected_model", modelName);
    router.push("/");
  };

  const formatSize = (bytes: number) => {
    return (bytes / (1024 * 1024 * 1024)).toFixed(1) + " GB";
  };

  const getRunnability = (reqRamGB: number) => {
    if (!sysRamBytes) return { label: "Unknown", color: "bg-gray-500/20 text-gray-400 border-gray-500/30" };
    if (reqRamGB === 0) return { label: "Unknown", color: "bg-gray-500/20 text-gray-400 border-gray-500/30" };
    
    const sysRamGB = sysRamBytes / (1024 * 1024 * 1024);
    if (sysRamGB >= reqRamGB * 1.5) {
      return { label: "Excellent", color: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30" };
    } else if (sysRamGB >= reqRamGB) {
      return { label: "Okay", color: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30" };
    } else {
      return { label: "Poor", color: "bg-red-500/20 text-red-400 border-red-500/30" };
    }
  };

  const parseParams = (paramStr: string) => {
    const val = parseFloat(paramStr.replace(/[^0-9.]/g, ""));
    if (isNaN(val)) return 0;
    if (paramStr.toLowerCase().includes("m")) return val / 1000;
    return val;
  };

  const allModels = [...customModels, ...CURATED_MODELS];

  const filteredModels = allModels.filter(
    (m) => m.name.toLowerCase().includes(search.toLowerCase()) || 
           m.description.toLowerCase().includes(search.toLowerCase())
  ).sort((a, b) => {
    if (sortBy === "name") {
      return a.name.localeCompare(b.name);
    } else if (sortBy === "params_asc") {
      return parseParams(a.params) - parseParams(b.params);
    } else if (sortBy === "params_desc") {
      return parseParams(b.params) - parseParams(a.params);
    }
    return 0;
  });

  const totalPages = Math.ceil(filteredModels.length / ITEMS_PER_PAGE);
  const paginatedModels = filteredModels.slice(
    (currentPage - 1) * ITEMS_PER_PAGE,
    currentPage * ITEMS_PER_PAGE
  );

  useEffect(() => {
    setCurrentPage(1); // Reset page on search or sort
  }, [search, sortBy]);

  return (
    <div className="flex flex-col h-full bg-[var(--background)]">
      {/* Header */}
      <div className="flex flex-col gap-4 p-6 border-b border-[var(--hover)] bg-[var(--color4)]">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-[var(--color2)] flex items-center gap-2">
              <Server className="text-blue-500" />
              Ollama Library
            </h1>
            <p className="text-[var(--color3)] mt-1 text-sm">
              Discover, download, and manage local AI models. 
            </p>
          </div>
          
          <button 
            onClick={() => setShowModal(true)}
            className="hidden sm:flex items-center gap-1.5 px-3 py-2 bg-[var(--hover)] text-[var(--color2)] rounded-lg text-sm font-medium border border-gray-700 hover:bg-gray-700 transition-colors"
          >
            <Plus size={16} /> Model Not Listed?
          </button>
        </div>
        
        <div className="flex flex-col md:flex-row items-start md:items-center gap-4 justify-between mt-2">
          <div className="flex flex-col sm:flex-row gap-3 w-full max-w-3xl">
            <div className="relative w-full max-w-md">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Search className="h-4 w-4 text-[var(--color3)]" />
              </div>
              <input
                type="text"
                placeholder="Search library..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="block w-full pl-9 pr-3 py-2 text-sm border border-[var(--hover)] rounded-xl bg-[var(--input-bg)] text-[var(--foreground)] placeholder-[var(--color3)] focus:outline-none focus:border-gray-500 transition-colors"
              />
            </div>
            
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="px-3 py-2 text-sm border border-[var(--hover)] rounded-xl bg-[var(--input-bg)] text-[var(--foreground)] focus:outline-none focus:border-gray-500 transition-colors cursor-pointer outline-none"
            >
              <option value="name">Sort by Name</option>
              <option value="params_asc">Sort by Parameters (Low &rarr; High)</option>
              <option value="params_desc">Sort by Parameters (High &rarr; Low)</option>
            </select>
            
            <button 
              onClick={() => setShowModal(true)}
              className="sm:hidden flex items-center justify-center gap-1.5 px-3 py-2 bg-[var(--hover)] text-[var(--color2)] rounded-xl text-sm font-medium border border-gray-700 hover:bg-gray-700 transition-colors w-full"
            >
              <Plus size={16} /> Model Not Listed?
            </button>
          </div>
          
          <div className="flex items-center gap-2 text-sm text-[var(--color3)] bg-[var(--hover)] px-3 py-1.5 rounded-lg border border-gray-700 shadow-sm whitespace-nowrap">
            <Cpu size={14} />
            <span>RAM: {sysRamBytes ? formatSize(sysRamBytes) : "..."}</span>
          </div>
        </div>
      </div>

      {/* Main Content (List View) */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-6 lg:px-12 xl:px-24">
        {loading ? (
          <div className="flex justify-center items-center h-40">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          </div>
        ) : paginatedModels.length === 0 ? (
          <div className="text-center text-[var(--color3)] mt-10">
            No models found matching &quot;{search}&quot;.
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            {paginatedModels.map((model) => {
              const installedInfo = installedModels.find(im => im.name === model.name);
              const isDownloaded = !!installedInfo;
              const runnability = getRunnability(model.reqRamGB);
              const isDownloading = downloading === model.name;
              
              return (
                <div 
                  key={model.name} 
                  className="bg-[var(--color4)] rounded-xl border border-[var(--hover)] p-4 flex flex-col md:flex-row md:items-center gap-4 hover:border-gray-600 transition-colors"
                >
                  {/* Info Section */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="text-base font-semibold text-[var(--color2)] truncate">
                        {model.name}
                      </h3>
                      {isDownloaded && (
                        <span className="flex items-center gap-1 text-xs px-2 py-0.5 bg-emerald-500/10 text-emerald-400 rounded-md border border-emerald-500/20">
                          <CheckCircle2 size={12} /> Installed
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-[var(--color3)] truncate max-w-2xl mb-3">
                      {model.description}
                    </p>
                    
                    {/* Tags */}
                    <div className="flex flex-wrap items-center gap-2">
                      <div className={`px-2 py-0.5 rounded text-[11px] font-semibold border ${runnability.color}`}>
                        {runnability.label}
                      </div>
                      <div className="flex items-center gap-1 px-2 py-0.5 bg-[var(--hover)] text-gray-400 rounded text-[11px] font-medium border border-gray-700">
                        <Hash size={12} /> {installedInfo?.details?.parameter_size || model.params}
                      </div>
                      <div className="flex items-center gap-1 px-2 py-0.5 bg-[var(--hover)] text-gray-400 rounded text-[11px] font-medium border border-gray-700">
                        <Database size={12} /> {model.reqRamGB ? `Req: ~${model.reqRamGB}GB` : 'Req: Unknown'}
                      </div>
                      {isDownloaded && installedInfo?.size && (
                        <div className="flex items-center gap-1 px-2 py-0.5 bg-[var(--hover)] text-gray-400 rounded text-[11px] font-medium border border-gray-700">
                          Disk: {formatSize(installedInfo.size)}
                        </div>
                      )}
                    </div>
                  </div>
                  
                  {/* Action Section */}
                  <div className="flex flex-row md:flex-col items-center justify-end gap-2 shrink-0 border-t md:border-t-0 md:border-l border-[var(--hover)] pt-3 md:pt-0 md:pl-4">
                    {isDownloaded ? (
                      <button 
                        onClick={() => handleUseModel(model.name)}
                        className="w-full md:w-32 px-4 py-2 rounded-lg bg-gray-200 hover:bg-white text-black text-sm font-medium transition-colors flex items-center justify-center gap-1"
                      >
                        Use <ChevronRight size={16} />
                      </button>
                    ) : (
                      <button 
                        onClick={() => handleDownload(model.name)}
                        disabled={!!downloading}
                        className={`w-full md:w-32 px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-1.5 ${
                          isDownloading 
                            ? "bg-blue-500/20 text-blue-400 cursor-not-allowed" 
                            : "bg-blue-600 hover:bg-blue-500 text-white shadow-md shadow-blue-900/20"
                        }`}
                      >
                        {isDownloading ? (
                          <>
                            <Loader2 size={16} className="animate-spin" /> Pulling
                          </>
                        ) : (
                          <>
                            <Download size={16} /> Download
                          </>
                        )}
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-center gap-2 mt-8">
            <button
              onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="px-3 py-1.5 rounded-lg border border-[var(--hover)] bg-[var(--color4)] text-[var(--color2)] text-sm disabled:opacity-50 hover:bg-[var(--hover)] transition-colors"
            >
              Previous
            </button>
            <span className="text-sm text-[var(--color3)] px-2">
              Page {currentPage} of {totalPages}
            </span>
            <button
              onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              className="px-3 py-1.5 rounded-lg border border-[var(--hover)] bg-[var(--color4)] text-[var(--color2)] text-sm disabled:opacity-50 hover:bg-[var(--hover)] transition-colors"
            >
              Next
            </button>
          </div>
        )}
      </div>

      {/* Custom Model Modal */}
      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
          <div className="bg-[var(--color4)] border border-[var(--hover)] rounded-2xl p-6 w-full max-w-md shadow-2xl relative">
            <button 
              onClick={() => setShowModal(false)}
              className="absolute top-4 right-4 text-[var(--color3)] hover:text-white"
            >
              <X size={20} />
            </button>
            
            <h2 className="text-xl font-bold text-white mb-2">Add Custom Model</h2>
            <p className="text-sm text-[var(--color3)] mb-6">
              Enter any valid Ollama model tag from the registry (e.g. <code>dolphin-mixtral:latest</code>). It will be saved to your library and downloaded immediately.
            </p>
            
            <input
              type="text"
              placeholder="Model tag (e.g. llama3:70b)"
              value={customModelName}
              onChange={(e) => setCustomModelName(e.target.value)}
              className="w-full px-4 py-2.5 bg-[var(--input-bg)] border border-[var(--hover)] rounded-xl text-white mb-4 focus:outline-none focus:border-blue-500"
              autoFocus
            />
            
            <div className="flex gap-3 justify-end">
              <button 
                onClick={() => setShowModal(false)}
                className="px-4 py-2 text-sm font-medium text-[var(--color3)] hover:text-white"
              >
                Cancel
              </button>
              <button 
                onClick={handleAddCustomModel}
                disabled={!customModelName.trim()}
                className="px-4 py-2 text-sm font-medium bg-blue-600 hover:bg-blue-500 text-white rounded-xl shadow-md disabled:opacity-50"
              >
                Add & Download
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
