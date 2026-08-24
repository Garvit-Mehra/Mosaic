"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { ArrowUp, ArrowDown, File as FileIcon, X, Loader2 } from "lucide-react";
import { useSession } from "next-auth/react";
import MessageBubble from "./components/chat/MessageBubble";
import ModelSelector from "./components/common/ModelSelector";
import FileUploadButton from "./components/chat/FileUploadButton";
import ActiveDocuments from "./components/chat/ActiveDocuments";

interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
  agent?: string;
  error?: boolean;
}

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL;

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState<number | null>(null);
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [uploadingFiles, setUploadingFiles] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { data: session } = useSession();
  const backendToken = (session as any)?.backendToken;

  // Auto-scroll (only when enabled)
  useEffect(() => {
    if (autoScroll) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, autoScroll]);

  // Detect manual scroll — disable auto-scroll if user scrolls up
  const handleScroll = useCallback(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const isAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 100;
    setAutoScroll(isAtBottom);
  }, []);

  // Auto-resize textarea
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

    

    // If not retry, add user message
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
      const convId = sessionStorage.getItem("tempConvId") || "temp_" + Date.now();
      sessionStorage.setItem("tempConvId", convId);
      
      for (const file of pendingFiles) {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("conversation_id", convId);

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
          temp_id: sessionStorage.getItem("tempConvId"),
        }),
      });

      if (res.status === 401) {
        window.location.href = "/login";
        return;
      }

      if (!res.ok) {
        let errorText = "An error occurred on the server.";
        if (res.status === 429) {
          errorText = "Too many messages, please slow down.";
        } else {
          try {
            const errData = await res.json();
            if (errData.detail) {
              errorText = typeof errData.detail === 'string' ? errData.detail : JSON.stringify(errData.detail);
            }
          } catch {
            // Ignore parse errors for non-JSON responses
          }
        }
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantId
              ? { ...msg, content: errorText, error: true }
              : msg
          )
        );
        setLoading(false);
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
            } else if (event.type === "done" && event.conversation_id) {
              if (!conversationId) {
                setConversationId(event.conversation_id);
                // Dispatch event to instantly refresh the sidebar
                window.dispatchEvent(new Event("mosaic-sidebar-refresh"));
              }
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
    // Find the user message before this assistant message
    const userMsg = messages.slice(0, messageIndex).reverse().find((m) => m.role === "user");
    if (!userMsg) return;

    // Remove the failed assistant message
    setMessages((prev) => prev.filter((_, i) => i !== messageIndex));

    // Resend
    sendMessage(userMsg.content);
  };

  // Auto-generate title after 3 messages
  useEffect(() => {
    if (messages.length === 3 && conversationId && backendToken) {
      const userMessages = messages.filter((m) => m.role === "user").map((m) => m.content).join(" | ");
      // Fire and forget — don't block the UI
      fetch(`${BACKEND}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${backendToken}`,
        },
          body: JSON.stringify({
          message: `Generate a short title (max 5 words) for a conversation about: "${userMessages}". Reply with ONLY the title, no quotes.`,
          conversation_id: null, // Don't pollute the actual conversation
          transient: true, // Don't create a new conversation in the backend DB
        }),
      })
        .then((r) => r.json())
        .then((data) => {
          if (data.response) {
            const title = data.response.trim().replace(/['"]/g, "").slice(0, 50);
            // Update the conversation title
            fetch(`${BACKEND}/conversations/${conversationId}`, {
              method: "PATCH",
              headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${backendToken}`,
              },
              body: JSON.stringify({ title }),
            });
          }
        })
        .catch(() => {}); // silent
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages.length]);

  // Handle New Chat event to reset state if we're already on /
  useEffect(() => {
    const handleNewChat = () => {
      setMessages([]);
      setInput("");
      setConversationId(null);
      // Clear temp documents
      const tempId = sessionStorage.getItem("tempConvId") || "temp";
      fetch(`${BACKEND}/api/documents/clear/${tempId}`, { method: "DELETE" })
        .catch(() => {})
        .finally(() => {
          sessionStorage.removeItem("tempConvId");
          window.dispatchEvent(new Event("document-uploaded"));
        });
    };
    window.addEventListener("mosaic-new-chat", handleNewChat);
    return () => window.removeEventListener("mosaic-new-chat", handleNewChat);
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-full relative">
      <ActiveDocuments conversationId={conversationId?.toString() || "temp"} />
      
      {/* Messages area */}
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto px-4 py-6"
      >
        <div className="max-w-3xl mx-auto space-y-4">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-[60vh] gap-3">
              <h1 className="text-4xl font-semibold text-[var(--color2)]">
                Mosaic
              </h1>
              <p className="text-[var(--color3)] text-sm italic">
                A modular multi-agent AI assistant
              </p>
            </div>
          )}

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

      {/* Scroll-to-bottom button (when auto-scroll is off) */}
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
      <div className={messages.length === 0 ? "flex-1 flex flex-col justify-center px-4" : "border-t border-transparent px-4 py-4"}>

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
