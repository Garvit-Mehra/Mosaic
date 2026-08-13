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

const extractText = (node: any): string => {
  if (node === null || node === undefined) return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(extractText).join("");
  if (node.props && node.props.children) {
    return extractText(node.props.children);
  }
  return "";
};

function CodeBlockWrapper({ language, rawCode, children }: any) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(rawCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative flex flex-col rounded-xl overflow-hidden my-3 border border-[var(--hover)] bg-[#0d1117] font-sans not-prose">
      <div className="flex items-center justify-between px-4 py-2 bg-[#161b22] text-[var(--color3)] text-xs border-b border-[var(--hover)]">
        <span className="font-mono">{language}</span>
        <button
          onClick={handleCopy}
          className="group/copy flex items-center gap-1.5 hover:text-white transition-colors"
        >
          {copied ? <Check className="w-3.5 h-3.5 text-green-400" /> : <Copy className="w-3.5 h-3.5" />}
          <span>{copied ? "Copied!" : "Copy"}</span>
        </button>
      </div>
      <div className="p-4 overflow-x-auto text-sm bg-[#0d1117]">
        <pre className="!bg-transparent !p-0 !m-0 !border-0 whitespace-pre">
          {children}
        </pre>
      </div>
    </div>
  );
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
          <div className="prose prose-invert prose-sm max-w-none [&_code]:text-xs [&_p]:my-1 [&_ul]:my-1 [&_ol]:my-1 [&_li]:my-0.5 [&_h1]:text-base [&_h2]:text-sm [&_h3]:text-sm [&_table]:text-xs [&_th]:px-2 [&_td]:px-2 [&_a]:text-[var(--color2)]">
            <ReactMarkdown 
              remarkPlugins={[remarkGfm]} 
              rehypePlugins={[rehypeHighlight]}
              components={{
                pre({ children }: any) {
                  const codeElement = Array.isArray(children) ? children[0] : children;
                  const className = codeElement?.props?.className || "";
                  const match = /language-(\w+)/.exec(className);
                  const language = match ? match[1] : "";
                  const rawCode = extractText(codeElement?.props?.children);
                  
                  return (
                    <CodeBlockWrapper language={language} rawCode={rawCode}>
                      {children}
                    </CodeBlockWrapper>
                  );
                },
                code({ inline, className, children, ...props }: any) {
                  if (inline) {
                    return (
                      <code className={`${className || ""} bg-[var(--hover)] px-1.5 py-0.5 rounded text-xs`} {...props}>
                        {children}
                      </code>
                    );
                  }
                  return (
                    <code className={className} {...props}>
                      {children}
                    </code>
                  );
                }
              }}
            >
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
