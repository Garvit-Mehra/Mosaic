"use client";

import { useState, useEffect } from "react";
import { FileText, X, Loader2, ChevronDown, ChevronUp } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Document {
  filename: string;
  size_chars: number;
}

export default function ActiveDocuments({ conversationId }: { conversationId: string | null }) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [isOpen, setIsOpen] = useState(false);
  const [deletingFile, setDeletingFile] = useState<string | null>(null);

  const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8080";

  const fetchDocuments = async () => {
    if (!conversationId) {
      setDocuments([]);
      setLoading(false);
      return;
    }
    try {
      const res = await fetch(`${BACKEND}/api/documents/${conversationId}`);
      if (res.ok) {
        const data = await res.json();
        setDocuments(data.documents || []);
      }
    } catch (e) {
      console.error("Failed to fetch documents", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();

    // Listen for custom event from FileUploadButton
    const handleUploadEvent = () => {
      fetchDocuments();
    };
    window.addEventListener("document-uploaded", handleUploadEvent);
    return () => window.removeEventListener("document-uploaded", handleUploadEvent);
  }, [conversationId]);

  const removeDocument = async (filename: string) => {
    if (!conversationId) return;
    setDeletingFile(filename);
    try {
      const res = await fetch(`${BACKEND}/api/documents/${conversationId}/${encodeURIComponent(filename)}`, {
        method: "DELETE",
      });
      if (res.ok) {
        setDocuments((prev) => prev.filter((d) => d.filename !== filename));
      }
    } catch (e) {
      console.error("Failed to delete document", e);
    } finally {
      setDeletingFile(null);
    }
  };

  if (!conversationId) return null;
  if (!loading && documents.length === 0) return null;

  return (
    <div className="absolute top-4 right-4 z-10 flex flex-col items-end">
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 mosaic-glass-button px-4 py-2 rounded-full text-sm font-medium text-[var(--color1)] transition-colors shadow-lg"
      >
        <FileText className="w-4 h-4 text-emerald-400" />
        <span>{documents.length} file{documents.length !== 1 ? 's' : ''}</span>
        {isOpen ? <ChevronUp className="w-4 h-4 ml-1 opacity-70" /> : <ChevronDown className="w-4 h-4 ml-1 opacity-70" />}
      </motion.button>

      <AnimatePresence>
      {isOpen && (
        <motion.div 
          initial={{ opacity: 0, y: -10, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -10, scale: 0.95 }}
          transition={{ type: "spring", stiffness: 400, damping: 30 }}
          className="mt-3 w-64 mosaic-panel rounded-2xl p-2 flex flex-col gap-1 max-h-64 overflow-y-auto origin-top-right"
        >
          {loading ? (
            <div className="flex justify-center p-2"><Loader2 className="w-4 h-4 animate-spin text-[var(--color1)] opacity-70" /></div>
          ) : (
            documents.map((doc) => (
              <motion.div 
                key={doc.filename}
                layout
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="flex items-center justify-between gap-2 p-3 rounded-2xl bg-[rgba(255,255,255,0.05)] border border-[rgba(255,255,255,0.1)] hover:bg-[rgba(255,255,255,0.1)] transition-colors"
              >
                <div className="flex flex-col overflow-hidden">
                  <span className="text-[13px] text-[var(--color1)] font-medium truncate" title={doc.filename}>{doc.filename}</span>
                  <span className="text-[11px] text-[var(--color1)] opacity-60">{Math.round(doc.size_chars / 1000)}k chars</span>
                </div>
                <button
                  onClick={() => removeDocument(doc.filename)}
                  disabled={deletingFile === doc.filename}
                  className="p-1.5 rounded-full bg-red-500/20 text-red-400 hover:bg-red-500/40 transition-colors"
                >
                  {deletingFile === doc.filename ? (
                    <Loader2 className="w-3 h-3 animate-spin" />
                  ) : (
                    <X className="w-3 h-3" />
                  )}
                </button>
              </motion.div>
            ))
          )}
        </motion.div>
      )}
      </AnimatePresence>
    </div>
  );
}
