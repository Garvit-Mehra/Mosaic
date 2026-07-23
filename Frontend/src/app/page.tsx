"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { ArrowUp, ArrowDown } from "lucide-react";
import { useSession } from "next-auth/react";
import MessageBubble from "./components/chat/MessageBubble";

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
    const messageContent = retryContent || input.trim();
    if (!messageContent || loading) return;

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
            } else if (event.type === "done" && event.conversation_id) {
              if (!conversationId) {
                setConversationId(event.conversation_id);
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
  }, [messages.length]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-full">
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
      <div className="border-t border-[var(--hover)] px-4 py-4">
        <div className="max-w-3xl mx-auto">
          <div className="flex items-end gap-2 bg-[var(--input-bg)] rounded-2xl p-3 border border-[var(--hover)]">
            <textarea
              ref={textareaRef}
              rows={1}
              className="flex-1 resize-none bg-transparent text-[var(--foreground)] placeholder-[var(--color3)] focus:outline-none text-sm px-2 py-1"
              placeholder="Ask Mosaic anything..."
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
