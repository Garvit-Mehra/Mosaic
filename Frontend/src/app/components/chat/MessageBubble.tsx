"use client";

import { useState } from "react";
import { Copy, Check, RotateCcw } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";

interface MessageBubbleProps {
  role: "user" | "assistant";
  content: string;
  agent?: string;
  isStreaming?: boolean;
  onRetry?: () => void;
  showRetry?: boolean;
}

export default function MessageBubble({
  role,
  content,
  agent,
  isStreaming,
  onRetry,
  showRetry,
}: MessageBubbleProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={`flex ${role === "user" ? "justify-end" : "justify-start"} group`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed relative ${
          role === "user"
            ? "bg-[var(--user-bubble)] text-[var(--color1)]"
            : "text-[var(--color1)]"
        }`}
      >
        {/* Content */}
        {role === "assistant" && content ? (
          <div className="prose prose-invert prose-sm max-w-none [&_pre]:bg-[var(--color4)] [&_pre]:border [&_pre]:border-[var(--hover)] [&_pre]:rounded-xl [&_pre]:p-3 [&_pre]:overflow-x-auto [&_code]:text-xs [&_p]:my-1 [&_ul]:my-1 [&_ol]:my-1 [&_li]:my-0.5 [&_h1]:text-base [&_h2]:text-sm [&_h3]:text-sm [&_table]:text-xs [&_th]:px-2 [&_td]:px-2 [&_a]:text-[var(--color2)]">
            <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
              {content}
            </ReactMarkdown>
          </div>
        ) : (
          <span className="whitespace-pre-wrap">{content}</span>
        )}

        {/* Streaming indicator */}
        {isStreaming && !content && (
          <span className="inline-flex items-center gap-1.5 py-1">
            <span className="typing-dot" />
            <span className="typing-dot" />
            <span className="typing-dot" />
          </span>
        )}

        {/* Agent badge */}
        {agent && role === "assistant" && content && !isStreaming && (
          <span className="block text-xs text-[var(--color3)] mt-1.5 opacity-60">
            via {agent}
          </span>
        )}

        {/* Action buttons (assistant only, not while streaming) */}
        {role === "assistant" && content && !isStreaming && (
          <div className="flex items-center gap-1 mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              onClick={handleCopy}
              className="p-1 rounded hover:bg-[var(--hover)] text-[var(--color3)] hover:text-[var(--color1)] transition-colors"
              title="Copy message"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-green-400" /> : <Copy className="w-3.5 h-3.5" />}
            </button>
            {showRetry && onRetry && (
              <button
                onClick={onRetry}
                className="p-1 rounded hover:bg-[var(--hover)] text-[var(--color3)] hover:text-[var(--color1)] transition-colors"
                title="Retry"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
