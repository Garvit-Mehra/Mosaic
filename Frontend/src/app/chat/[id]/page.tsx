"use client";

import { useState, useRef, useEffect, useCallback, use } from "react";
import { ArrowUp, ArrowDown, Plus, File as FileIcon, X, Loader2 } from "lucide-react";
import { useSession } from "next-auth/react";
import { authFetch } from "@/src/lib/auth";
import MessageBubble from "@/src/app/components/chat/MessageBubble";
import ModelSelector from "../../components/common/ModelSelector";
import FileUploadButton from "@/src/app/components/chat/FileUploadButton";
import ActiveDocuments from "@/src/app/components/chat/ActiveDocuments";

interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
  agent?: string;
  error?: boolean;
}

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL;

export default function ConversationPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const conversationId = parseInt(id);

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [uploadingFiles, setUploadingFiles] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { data: session } = useSession();
  const backendToken = (session as any)?.backendToken;

  // Load conversation history
  useEffect(() => {
    const loadConversation = async () => {
      if (!backendToken) return;
      try {
        const res = await authFetch(`${BACKEND}/conversations/${conversationId}`, {}, backendToken);
        if (res.ok) {
          const data = await res.json();
          setMessages(
            data.messages.map((m: { role: string; content: string; agent?: string }, i: number) => ({
              id: i,
              role: m.role as "user" | "assistant",
              content: m.content,
              agent: m.agent,
            }))
          );
        }
      } catch {
        // Backend not reachable
      } finally {
        setLoadingHistory(false);
      }
    };
    loadConversation();
  }, [conversationId, backendToken]);

  useEffect(() => {
    if (autoScroll) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, autoScroll]);

  const handleScroll = useCallback(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const isAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 100;
    setAutoScroll(isAtBottom);
  }, []);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, 200) + "px";
    }
  }, [input]);

  const sendMessage = async (retryContent?: string) => {
    let messageContent = retryContent || input.trim();
    
    // Prepend attached files info if this is a new message with files
    if (!retryContent && pendingFiles.length > 0) {
      const fileNames = pendingFiles.map(f => f.name).join(", ");
      const prefix = `[Attached files: ${fileNames}]\n\n`;
      messageContent = prefix + (messageContent || "Please analyze the attached document(s).");
    }
    if (!messageContent && pendingFiles.length === 0) return;
    if (loading) return;

    

    if (!retryContent) {
      const userMessage: Message = {
        id: Date.now(),
        role: "user",
        content: messageContent,
      };
      setMessages((prev) => [...prev, userMessage]);
      setInput("");
    }

    setLoading(true);
    setAutoScroll(true);

    // Upload pending files first
    if (pendingFiles.length > 0) {
      setUploadingFiles(true);
      const convId = conversationId || "temp";
      
      for (const file of pendingFiles) {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("conversation_id", convId.toString());

        try {
          await fetch(`${BACKEND}/api/documents/upload`, {
            method: "POST",
            body: formData,
          });
        } catch (e) {
          console.error("Failed to upload file", e);
        }
      }
      
      setPendingFiles([]);
      setUploadingFiles(false);
      window.dispatchEvent(new Event("document-uploaded"));
    }

    const assistantId = Date.now() + 1;
    setMessages((prev) => [
      ...prev,
      { id: assistantId, role: "assistant", content: "", agent: undefined },
    ]);

    try {
      const selectedModel = localStorage.getItem("mosaic_selected_model") || "llama3.2";

      const res = await fetch(`${BACKEND}/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(backendToken ? { "Authorization": `Bearer ${backendToken}` } : {}),
        },
        body: JSON.stringify({
          message: messageContent,
          conversation_id: conversationId,
          model: selectedModel,
        }),
      });

      if (res.status === 401) {
        window.location.href = "/login";
        return;
      }

      if (!res.body) throw new Error("No response body");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const jsonStr = line.slice(6);
          try {
            const event = JSON.parse(jsonStr);

            if (event.type === "token") {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantId
                    ? { ...msg, content: msg.content + event.content }
                    : msg
                )
              );
            } else if (event.type === "agent") {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantId
                    ? { ...msg, agent: event.agent }
                    : msg
                )
              );
            } else if (event.type === "error") {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantId
                    ? { ...msg, content: event.content, error: true }
                    : msg
                )
              );
            }
          } catch {
            // Skip malformed JSON
          }
        }
      }
    } catch {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantId
            ? { ...msg, content: "Could not connect to the backend. Is it running?", error: true }
            : msg
        )
      );
    } finally {
      setLoading(false);
    }
  };

  const handleRetry = (messageIndex: number) => {
    const userMsg = messages.slice(0, messageIndex).reverse().find((m) => m.role === "user");
    if (!userMsg) return;
    setMessages((prev) => prev.filter((_, i) => i !== messageIndex));
    sendMessage(userMsg.content);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  if (loadingHistory) {
    return (
      <div className="flex flex-col h-full">
        <div className="flex-1 overflow-y-auto px-4 py-6">
          <div className="max-w-3xl mx-auto space-y-6">
            <div className="flex justify-end">
              <div className="w-1/3 h-10 bg-[var(--hover)] rounded-2xl animate-pulse" />
            </div>
            <div className="flex justify-start">
              <div className="w-2/3 h-20 bg-[var(--hover)] rounded-2xl animate-pulse" />
            </div>
            <div className="flex justify-end">
              <div className="w-1/2 h-16 bg-[var(--hover)] rounded-2xl animate-pulse" />
            </div>
          </div>
        </div>
        <div className="border-t border-[var(--hover)] px-4 py-4">
          <div className="max-w-3xl mx-auto">
            <div className="h-12 bg-[var(--input-bg)] rounded-2xl border border-[var(--hover)] animate-pulse" />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full relative">
      <ActiveDocuments conversationId={String(conversationId)} />
      {/* Messages */}
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto px-4 py-6"
      >
        <div className="max-w-3xl mx-auto space-y-4">
          {messages.map((msg, index) => (
            <MessageBubble
              key={msg.id}
              role={msg.role}
              content={msg.content}
              agent={msg.agent}
              isStreaming={loading && msg.role === "assistant" && index === messages.length - 1 && !msg.content}
              showRetry={msg.error}
              onRetry={() => handleRetry(index)}
            />
          ))}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* Scroll-to-bottom button */}
      {!autoScroll && messages.length > 0 && (
        <div className="absolute bottom-24 left-1/2 -translate-x-1/2">
          <button
            onClick={() => {
              setAutoScroll(true);
              bottomRef.current?.scrollIntoView({ behavior: "smooth" });
            }}
            className="p-2 rounded-full bg-[var(--input-bg)] border border-[var(--hover)] text-[var(--color3)] hover:text-[var(--color1)] shadow-lg transition-colors"
          >
            <ArrowDown className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Input area */}
      <div className="border-t border-[var(--hover)] px-4 py-4">
        <div className="max-w-3xl mx-auto w-full">
          <div className="flex flex-col gap-2 bg-[#2f2f2f] rounded-3xl p-2 border border-transparent focus-within:border-gray-600 transition-colors shadow-sm">
            {/* Pending Files Tray */}
            {pendingFiles.length > 0 && (
              <div className="flex flex-wrap gap-2 px-3 pt-2">
                {pendingFiles.map((file, idx) => (
                  <div key={idx} className="flex items-center gap-2 bg-[#404040] rounded-xl px-3 py-2 text-sm max-w-[200px]">
                    {uploadingFiles ? (
                      <Loader2 className="w-4 h-4 animate-spin text-[var(--color1)] shrink-0" />
                    ) : (
                      <FileIcon className="w-4 h-4 text-[var(--color1)] shrink-0" />
                    )}
                    <span className="truncate text-gray-200">{file.name}</span>
                    {!uploadingFiles && (
                      <button
                        onClick={() => setPendingFiles(prev => prev.filter((_, i) => i !== idx))}
                        className="p-0.5 hover:bg-[#505050] rounded-full text-gray-400 hover:text-white shrink-0"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}

            <textarea
              ref={textareaRef}
              rows={1}
              className="flex-1 w-full resize-none bg-transparent text-gray-200 placeholder-gray-400 focus:outline-none text-base px-4 py-3 min-h-[52px]"
              placeholder="Send a message"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
            />
            
            <div className="flex items-center justify-end gap-2 px-2 pb-1">
              <FileUploadButton onFileSelect={(file) => setPendingFiles(prev => [...prev, file])} disabled={uploadingFiles} />
              
              <ModelSelector />
              
              <button
                onClick={() => sendMessage()}
                disabled={(!input.trim() && pendingFiles.length === 0) || loading}
                className={`p-2 rounded-full transition-colors ${
                  (!input.trim() && pendingFiles.length === 0) || loading 
                  ? "bg-gray-600/30 text-gray-500 cursor-not-allowed" 
                  : "bg-white text-black hover:bg-gray-200 cursor-pointer"
                }`}
              >
                <ArrowUp className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
