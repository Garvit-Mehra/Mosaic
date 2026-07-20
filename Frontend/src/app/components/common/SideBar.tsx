"use client";

import { useState, useEffect } from "react";
import { MessageCircle, SquarePen, ChevronLeft, ChevronRight, Trash2, Settings, LogOut, Shield } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useSession, signOut } from "next-auth/react";
import { authFetch } from "@/src/lib/auth";

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL;

interface Conversation {
  id: number;
  title: string;
  updated_at: string;
}

export default function SideBar() {
  const [collapsed, setCollapsed] = useState(false);
  const [hovered, setHovered] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const pathname = usePathname();
  const { data: session } = useSession();

  const userIsAdmin = (session as any)?.role === "admin" || (session?.user as any)?.role === "admin";

  const fetchConversations = async () => {
    try {
      const res = await authFetch(`${BACKEND}/conversations`);
      if (res.ok) {
        const data = await res.json();
        setConversations(data);
      }
    } catch {
      // silent
    }
  };

  useEffect(() => {
    if (session) {
      fetchConversations();
      const interval = setInterval(fetchConversations, 10000);
      return () => clearInterval(interval);
    }
  }, [session]);

  const deleteConversation = async (id: number, e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    try {
      await authFetch(`${BACKEND}/conversations/${id}`, { method: "DELETE" });
      setConversations((prev) => prev.filter((c) => c.id !== id));
    } catch {
      // silent
    }
  };

  return (
    <aside
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="relative h-screen flex-shrink-0"
    >
      {/* Collapse toggle */}
      <button
        className={`flex justify-center items-center bg-[var(--color2)] text-[var(--color4)] cursor-pointer
          w-[22px] h-[36px] absolute top-7 right-[-22px] rounded-r-lg z-10
          transition-opacity duration-300 ease-in-out
          ${hovered ? "opacity-100" : "opacity-0 pointer-events-none"}`}
        onClick={() => setCollapsed(!collapsed)}
        onMouseEnter={() => setHovered(true)}
      >
        {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
      </button>

      <div
        className={`flex flex-col h-full transition-all duration-300 ease-in-out bg-[var(--color4)] border-r border-[var(--hover)] ${
          collapsed ? "w-14" : "w-64"
        }`}
      >
        {/* Header */}
        <div className="flex items-center px-3 py-4">
          <Link href="/" className="flex items-center gap-2">
            <span
              className={`font-bold text-[var(--color2)] transition-all duration-300 ${
                collapsed ? "text-lg" : "text-xl"
              }`}
            >
              {collapsed ? "M" : "Mosaic"}
            </span>
          </Link>
        </div>

        {/* New Chat button */}
        <div className="px-2 mb-2">
          <Link
            href="/"
            className={`flex items-center gap-2 px-3 py-2 rounded-xl text-sm hover:bg-[var(--hover)] transition-colors ${
              collapsed ? "justify-center" : ""
            }`}
          >
            <SquarePen size={16} className="text-[var(--color3)] flex-shrink-0" />
            {!collapsed && <span className="text-[var(--color3)]">New Chat</span>}
          </Link>
        </div>

        {/* Conversations list */}
        <div className="flex-1 overflow-y-auto px-2">
          {!collapsed && (
            <div className="px-3 py-1 text-xs text-[var(--color3)] uppercase tracking-wider">
              Chats
            </div>
          )}
          <ul className="space-y-0.5">
            {conversations.map((convo) => (
              <li key={convo.id} className="group">
                <Link
                  href={`/chat/${convo.id}`}
                  className={`flex items-center gap-2 px-3 py-2 rounded-xl text-sm transition-colors hover:bg-[var(--hover)] ${
                    pathname === `/chat/${convo.id}` ? "bg-[var(--hover)]" : ""
                  } ${collapsed ? "justify-center" : ""}`}
                >
                  <MessageCircle size={14} className="text-[var(--color3)] flex-shrink-0" />
                  {!collapsed && (
                    <>
                      <span className="flex-1 truncate text-[var(--color1)]">
                        {convo.title}
                      </span>
                      <button
                        onClick={(e) => deleteConversation(convo.id, e)}
                        className="opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:text-red-400"
                      >
                        <Trash2 size={12} />
                      </button>
                    </>
                  )}
                </Link>
              </li>
            ))}
          </ul>
        </div>

        {/* Bottom — Servers, Admin, Logout */}
        <div className="px-2 py-3 border-t border-[var(--hover)] space-y-1">
          <Link
            href="/settings"
            className={`flex items-center gap-2 px-3 py-2 rounded-xl text-sm hover:bg-[var(--hover)] transition-colors ${
              pathname === "/settings" ? "bg-[var(--hover)]" : ""
            } ${collapsed ? "justify-center" : ""}`}
          >
            <Settings size={16} className="text-[var(--color3)] flex-shrink-0" />
            {!collapsed && <span className="text-[var(--color3)]">Servers</span>}
          </Link>
          {userIsAdmin && (
            <Link
              href="/admin"
              className={`flex items-center gap-2 px-3 py-2 rounded-xl text-sm hover:bg-[var(--hover)] transition-colors ${
                pathname === "/admin" ? "bg-[var(--hover)]" : ""
              } ${collapsed ? "justify-center" : ""}`}
            >
              <Shield size={16} className="text-[var(--color3)] flex-shrink-0" />
              {!collapsed && <span className="text-[var(--color3)]">Admin</span>}
            </Link>
          )}
          <button
            onClick={() => signOut({ callbackUrl: "/login" })}
            className={`flex items-center gap-2 px-3 py-2 rounded-xl text-sm hover:bg-[var(--hover)] transition-colors w-full ${
              collapsed ? "justify-center" : ""
            }`}
          >
            <LogOut size={16} className="text-[var(--color3)] flex-shrink-0" />
            {!collapsed && <span className="text-[var(--color3)]">Logout</span>}
          </button>
        </div>
      </div>
    </aside>
  );
}
