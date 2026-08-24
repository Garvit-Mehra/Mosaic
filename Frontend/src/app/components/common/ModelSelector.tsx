"use client";

import React, { useState, useEffect } from "react";
import { ChevronDown, Loader2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

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
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-4 py-2 rounded-full mosaic-glass-button text-sm font-medium text-[var(--color1)] transition-colors"
      >
        <div className="flex items-center gap-1.5">
          {loading ? (
            <Loader2 className="w-4 h-4 animate-spin opacity-70" />
          ) : (
            selectedModel
          )}
        </div>
        <ChevronDown className="w-4 h-4 opacity-70" />
      </motion.button>

      <AnimatePresence>
      {isOpen && (
        <motion.div 
          initial={{ opacity: 0, y: 10, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 10, scale: 0.95 }}
          transition={{ type: "spring", stiffness: 400, damping: 30 }}
          className="absolute right-0 bottom-full mb-3 w-56 mosaic-panel rounded-2xl p-2 z-50 origin-bottom-right"
        >
          <div className="p-1">
            <div className="px-3 py-2 text-[10px] font-bold text-[var(--color3)] uppercase tracking-wider mb-1">
              Local Ollama Models
            </div>
            {models.length === 0 && !loading && (
              <div className="px-3 py-2 text-sm text-[var(--color3)]">No models found</div>
            )}
            <div className="space-y-1">
              {models.map((m) => (
                <button
                  key={m}
                  onClick={() => handleSelect(m)}
                  className={`w-full text-left px-3 py-2.5 text-sm rounded-xl transition-all ${
                    selectedModel === m
                      ? "bg-[rgba(255,255,255,0.2)] text-[var(--color1)] font-semibold shadow-[inset_0_1px_1px_rgba(255,255,255,0.1)]"
                      : "text-[var(--color1)] opacity-80 hover:opacity-100 hover:bg-[rgba(255,255,255,0.1)]"
                  }`}
                >
                  {m}
                </button>
              ))}
            </div>
          </div>
        </motion.div>
      )}
      </AnimatePresence>
    </div>
  );
}
