"use client";

import { useState, useRef, useEffect, useCallback, use } from "react";
import { ArrowUp, ArrowDown, Loader2 } from "lucide-react";
import { useSession } from "next-auth/react";
import { authFetch } from "@/src/lib/auth";
import MessageBubble from "@/src/app/components/chat/MessageBubble";

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
    const messageContent = retryContent || input.trim();
    if (!messageContent || loading) return;

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

    const assistantId = Date.now() + 1;
    setMessages((prev) => [
      ...prev,
      { id: assistantId, role: "assistant", content: "", agent: undefined },
    ]);

    try {
      const res = await fetch(`${BACKEND}/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(backendToken ? { "Authorization": `Bearer ${backendToken}` } : {}),
        },
        body: JSON.stringify({
          message: messageContent,
          conversation_id: conversationId,
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
      <div className="flex items-center justify-center h-full text-[var(--color3)]">
        <Loader2 className="w-6 h-6 animate-spin" />
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
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

      {/* Input */}
      <div className="border-t border-[var(--hover)] px-4 py-4">
        <div className="max-w-3xl mx-auto">
          <div className="flex items-end gap-2 bg-[var(--input-bg)] rounded-2xl p-3 border border-[var(--hover)]">
            <textarea
              ref={textareaRef}
              rows={1}
              className="flex-1 resize-none bg-transparent text-[var(--foreground)] placeholder-[var(--color3)] focus:outline-none text-sm px-2 py-1"
              placeholder="Continue the conversation..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
            />
            <button
              onClick={() => sendMessage()}
              disabled={!input.trim() || loading}
              className="p-2 rounded-xl bg-[var(--color2)] text-[var(--color4)] disabled:opacity-30 hover:opacity-80 transition-opacity cursor-pointer disabled:cursor-not-allowed"
            >
              <ArrowUp className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
