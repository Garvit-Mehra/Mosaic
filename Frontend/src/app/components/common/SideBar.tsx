"use client";

import { useState, useEffect, useMemo } from "react";
import { MessageCircle, SquarePen, ChevronLeft, ChevronRight, Trash2, Settings, LogOut, Shield, Sun, Moon, Search, Database } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useSession, signOut } from "next-auth/react";
import { authFetch } from "@/src/lib/auth";
import { useTheme } from "@/src/lib/theme";
import { motion, AnimatePresence } from "framer-motion";

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL;

interface Conversation {
  id: number;
  title: string;
  updated_at: string;
}

const butterySpring: any = {
  type: "spring",
  stiffness: 400,
  damping: 30,
  mass: 1,
  restDelta: 0.001
};

export default function SideBar() {
  const [collapsed, setCollapsed] = useState(false);
  const [hovered, setHovered] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const pathname = usePathname();
  const router = useRouter();
  const { data: session } = useSession();
  const { theme, toggle: toggleTheme } = useTheme();

  const userIsAdmin = (session as any)?.role === "admin" || (session?.user as any)?.role === "admin";
  const backendToken = (session as any)?.backendToken;

  const filteredConversations = useMemo(() => {
    if (!searchQuery.trim()) return conversations;
    const q = searchQuery.toLowerCase();
    return conversations.filter((c) => c.title.toLowerCase().includes(q));
  }, [conversations, searchQuery]);

  const fetchConversations = async () => {
    if (!backendToken) return;
    try {
      const res = await authFetch(`${BACKEND}/conversations`, {}, backendToken);
      if (res.ok) {
        const data = await res.json();
        setConversations(data);
      }
    } catch {}
  };

  useEffect(() => {
    if (backendToken) {
      fetchConversations();
      const interval = setInterval(fetchConversations, 10000);
      const handleRefresh = () => fetchConversations();
      window.addEventListener("mosaic-sidebar-refresh", handleRefresh);
      return () => {
        clearInterval(interval);
        window.removeEventListener("mosaic-sidebar-refresh", handleRefresh);
      };
    }
  }, [backendToken]);

  const deleteConversation = async (id: number, e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    try {
      await authFetch(`${BACKEND}/conversations/${id}`, { method: "DELETE" }, backendToken);
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (pathname === `/chat/${id}`) {
        router.push("/");
      }
    } catch {}
  };

  const itemVariants = {
    rest: { scale: 1 },
    hover: { scale: 1.03, backgroundColor: "rgba(255, 255, 255, 0.1)" },
    tap: { scale: 0.95 }
  };

  return (
    <aside
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="relative h-screen flex-shrink-0 py-[6px] pl-[6px] pr-0"
    >
      <motion.button
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
        className={`absolute top-10 right-[-10px] z-20 w-8 h-8 rounded-full flex items-center justify-center mosaic-panel text-[var(--color1)]
          transition-opacity duration-300 ease-in-out shadow-lg
          ${hovered ? "opacity-100" : "opacity-0 pointer-events-none"}`}
        onClick={() => setCollapsed(!collapsed)}
      >
        {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
      </motion.button>

      <motion.div
        animate={{ width: collapsed ? 80 : 280 }}
        transition={butterySpring}
        className="mosaic-panel h-full rounded-xl flex flex-col overflow-hidden"
      >
        {/* Header */}
        <div className="flex items-center px-5 py-6">
          <Link href="/" className="flex items-center gap-3">
            <motion.span layout transition={butterySpring} className="font-bold text-2xl text-[var(--color1)] tracking-tight">
              {collapsed ? "M" : "Mosaic"}
            </motion.span>
          </Link>
        </div>

        {/* New Chat */}
        <div className="px-3 mb-4">
          <motion.div variants={itemVariants} initial="rest" whileHover="hover" whileTap="tap" className="rounded-2xl">
            <Link
              href="/"
              onClick={(e) => {
                if (pathname === "/") {
                  e.preventDefault();
                  window.dispatchEvent(new Event("mosaic-new-chat"));
                }
              }}
              className={`flex items-center gap-3 px-4 py-3 rounded-2xl text-sm font-medium transition-colors ${collapsed ? "justify-center" : ""}`}
            >
              <SquarePen size={18} className="text-[var(--color1)] flex-shrink-0" />
              <AnimatePresence>
                {!collapsed && (
                  <motion.span initial={{ opacity: 0, width: 0 }} animate={{ opacity: 1, width: "auto" }} exit={{ opacity: 0, width: 0 }} className="text-[var(--color1)]">
                    New Chat
                  </motion.span>
                )}
              </AnimatePresence>
            </Link>
          </motion.div>
        </div>

        {/* Search + Chats */}
        <div className="flex-1 overflow-y-auto px-3">
          <AnimatePresence>
            {!collapsed && (
              <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}>
                <div className="px-1 py-2 mb-2">
                  <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-[var(--input-bg)] mosaic-input-focus border border-transparent transition-all">
                    <Search size={14} className="text-[var(--color3)] flex-shrink-0" />
                    <input
                      type="text"
                      placeholder="Search chats..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="bg-transparent text-sm text-[var(--color1)] placeholder-[var(--color3)] focus:outline-none w-full"
                    />
                  </div>
                </div>
                <div className="px-3 py-2 text-xs font-semibold text-[var(--color3)] uppercase tracking-wider">
                  Chats
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          
          <ul className="space-y-1">
            {filteredConversations.map((convo) => (
              <li key={convo.id} className="group">
                <motion.div variants={itemVariants} initial="rest" whileHover="hover" whileTap="tap" className="rounded-2xl">
                  <Link
                    href={`/chat/${convo.id}`}
                    className={`flex items-center gap-3 px-4 py-2.5 rounded-2xl text-sm font-medium transition-colors ${
                      pathname === `/chat/${convo.id}` ? "bg-[var(--hover)]" : ""
                    } ${collapsed ? "justify-center" : ""}`}
                  >
                    <MessageCircle size={16} className="text-[var(--color3)] flex-shrink-0" />
                    {!collapsed && (
                      <>
                        <span className="flex-1 truncate text-[var(--color1)]">{convo.title}</span>
                        <button
                          onClick={(e) => deleteConversation(convo.id, e)}
                          className="opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:text-red-400"
                        >
                          <Trash2 size={14} />
                        </button>
                      </>
                    )}
                  </Link>
                </motion.div>
              </li>
            ))}
          </ul>
        </div>

        {/* Bottom Options */}
        <div className="px-3 py-4 mt-auto border-t border-[var(--glass-border)] space-y-1">
          <motion.button variants={itemVariants} initial="rest" whileHover="hover" whileTap="tap"
            onClick={toggleTheme} className={`flex items-center gap-3 px-4 py-3 rounded-2xl text-sm font-medium w-full ${collapsed ? "justify-center" : ""}`}>
            {theme === "dark" ? <Sun size={18} className="text-[var(--color3)] flex-shrink-0" /> : <Moon size={18} className="text-[var(--color3)] flex-shrink-0" />}
            {!collapsed && <span className="text-[var(--color1)]">{theme === "dark" ? "Light mode" : "Dark mode"}</span>}
          </motion.button>
          
          <motion.div variants={itemVariants} initial="rest" whileHover="hover" whileTap="tap" className="rounded-2xl">
            <Link href="/models" className={`flex items-center gap-3 px-4 py-3 rounded-2xl text-sm font-medium ${pathname === "/models" ? "bg-[var(--hover)]" : ""} ${collapsed ? "justify-center" : ""}`}>
              <Database size={18} className="text-[var(--color3)] flex-shrink-0" />
              {!collapsed && <span className="text-[var(--color1)]">Models</span>}
            </Link>
          </motion.div>

          <motion.div variants={itemVariants} initial="rest" whileHover="hover" whileTap="tap" className="rounded-2xl">
            <Link href="/settings" className={`flex items-center gap-3 px-4 py-3 rounded-2xl text-sm font-medium ${pathname === "/settings" ? "bg-[var(--hover)]" : ""} ${collapsed ? "justify-center" : ""}`}>
              <Settings size={18} className="text-[var(--color3)] flex-shrink-0" />
              {!collapsed && <span className="text-[var(--color1)]">Servers</span>}
            </Link>
          </motion.div>

          {userIsAdmin && (
            <motion.div variants={itemVariants} initial="rest" whileHover="hover" whileTap="tap" className="rounded-2xl">
              <Link href="/admin" className={`flex items-center gap-3 px-4 py-3 rounded-2xl text-sm font-medium ${pathname === "/admin" ? "bg-[var(--hover)]" : ""} ${collapsed ? "justify-center" : ""}`}>
                <Shield size={18} className="text-[var(--color3)] flex-shrink-0" />
                {!collapsed && <span className="text-[var(--color1)]">Admin</span>}
              </Link>
            </motion.div>
          )}

          <motion.button variants={itemVariants} initial="rest" whileHover="hover" whileTap="tap"
            onClick={() => signOut({ callbackUrl: "/login" })} className={`flex items-center gap-3 px-4 py-3 rounded-2xl text-sm font-medium w-full ${collapsed ? "justify-center" : ""}`}>
            <LogOut size={18} className="text-[var(--color3)] flex-shrink-0" />
            {!collapsed && <span className="text-[var(--color1)]">Logout</span>}
          </motion.button>
        </div>
      </motion.div>
    </aside>
  );
}
