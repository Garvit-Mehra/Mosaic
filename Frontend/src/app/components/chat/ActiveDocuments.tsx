"use client";

import { useState, useEffect } from "react";
import { FileText, X, Loader2, ChevronDown, ChevronUp } from "lucide-react";

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
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 bg-[#2f2f2f] border border-[#404040] px-3 py-1.5 rounded-lg text-sm text-[var(--color3)] hover:text-white transition-colors shadow-lg"
      >
        <FileText className="w-4 h-4 text-emerald-400" />
        <span>{documents.length} file{documents.length !== 1 ? 's' : ''}</span>
        {isOpen ? <ChevronUp className="w-4 h-4 ml-1" /> : <ChevronDown className="w-4 h-4 ml-1" />}
      </button>

      {isOpen && (
        <div className="mt-2 w-64 bg-[#2f2f2f] border border-[#404040] rounded-xl shadow-2xl p-2 flex flex-col gap-1 max-h-64 overflow-y-auto">
          {loading ? (
            <div className="flex justify-center p-2"><Loader2 className="w-4 h-4 animate-spin text-gray-400" /></div>
          ) : (
            documents.map((doc) => (
              <div key={doc.filename} className="flex items-center justify-between gap-2 p-2 rounded-lg bg-[#202020] border border-[#303030]">
                <div className="flex flex-col overflow-hidden">
                  <span className="text-xs text-gray-200 truncate" title={doc.filename}>{doc.filename}</span>
                  <span className="text-[10px] text-gray-500">{Math.round(doc.size_chars / 1000)}k chars</span>
                </div>
                <button
                  onClick={() => removeDocument(doc.filename)}
                  disabled={deletingFile === doc.filename}
                  className="p-1 rounded bg-red-500/10 text-red-400 hover:bg-red-500/30 transition-colors"
                >
                  {deletingFile === doc.filename ? (
                    <Loader2 className="w-3 h-3 animate-spin" />
                  ) : (
                    <X className="w-3 h-3" />
                  )}
                </button>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
