"use client";

import React, { useState, useEffect } from "react";
import { ChevronDown, Loader2 } from "lucide-react";

export default function ModelSelector() {
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("llama3.2");
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(true);

  const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8080";

  useEffect(() => {
    // Load saved model
    const saved = localStorage.getItem("mosaic_selected_model");
    if (saved) {
      setSelectedModel(saved);
    }

    // Fetch models
    const fetchModels = async () => {
      try {
        const res = await fetch(`${BACKEND}/api/models`);
        if (res.ok) {
          const data = await res.json();
          setModels(data.models || ["llama3.2", "mistral"]);
          if (!saved && data.models && data.models.length > 0) {
            setSelectedModel(data.models[0]);
            localStorage.setItem("mosaic_selected_model", data.models[0]);
          }
        }
      } catch (e) {
        console.error("Failed to fetch models", e);
        setModels(["llama3.2", "mistral"]);
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
  }, [BACKEND]);

  const handleSelect = (model: string) => {
    setSelectedModel(model);
    localStorage.setItem("mosaic_selected_model", model);
    setIsOpen(false);
  };

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-gray-600/20 hover:bg-gray-600/40 text-sm font-medium text-gray-300 transition-colors"
      >
        <div className="flex items-center gap-1.5">
          {loading ? (
            <Loader2 className="w-3 h-3 animate-spin text-gray-400" />
          ) : (
            selectedModel
          )}
        </div>
        <ChevronDown className="w-4 h-4 text-gray-400" />
      </button>

      {isOpen && (
        <div className="absolute right-0 bottom-full mb-2 w-48 rounded-xl bg-[#2a2a2a] border border-gray-700 shadow-xl overflow-hidden z-50">
          <div className="p-1.5">
            <div className="px-2 py-1.5 text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">
              Local Ollama Models
            </div>
            {models.length === 0 && !loading && (
              <div className="px-2 py-2 text-sm text-gray-500">No models found</div>
            )}
            {models.map((m) => (
              <button
                key={m}
                onClick={() => handleSelect(m)}
                className={`w-full text-left px-3 py-2 text-sm rounded-lg transition-colors ${
                  selectedModel === m
                    ? "bg-blue-500/10 text-blue-400 font-medium"
                    : "text-gray-300 hover:bg-gray-800"
                }`}
              >
                {m}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
