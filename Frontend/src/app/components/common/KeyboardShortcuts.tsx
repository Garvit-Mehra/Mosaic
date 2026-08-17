"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function KeyboardShortcuts() {
  const router = useRouter();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Cmd+K or Ctrl+K for New Chat
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        if (window.location.pathname === "/") {
          window.dispatchEvent(new Event("mosaic-new-chat"));
        } else {
          router.push("/");
        }
      }
      
      // Cmd+/ or Ctrl+/ for Settings
      if ((e.metaKey || e.ctrlKey) && e.key === "/") {
        e.preventDefault();
        router.push("/settings");
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [router]);

  return null;
}
