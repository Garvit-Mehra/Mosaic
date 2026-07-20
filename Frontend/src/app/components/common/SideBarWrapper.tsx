"use client";

import { usePathname } from "next/navigation";
import SideBar from "./SideBar";

export default function SideBarWrapper() {
  const pathname = usePathname();

  // Don't show sidebar on login page
  if (pathname === "/login") return null;

  return <SideBar />;
}
