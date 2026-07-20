/**
 * Mosaic Auth Client Utilities
 * 
 * Works with NextAuth v5 sessions (httpOnly cookies).
 * No localStorage, no token exposure to JavaScript — XSS-safe.
 * 
 * For components that need session data, use the useSession() hook from next-auth/react.
 * For server components, use auth() from @/src/auth.
 * 
 * This file provides the authFetch() helper for making authenticated API calls
 * to the Python backend.
 */

import { getSession } from "next-auth/react";

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL;

export interface User {
  name?: string | null;
  email?: string | null;
  image?: string | null;
  role?: "admin" | "user";
}

/**
 * Make an authenticated fetch to the backend.
 * Retrieves the backend token from the NextAuth session and includes it as Bearer.
 */
export async function authFetch(url: string, options: RequestInit = {}): Promise<Response> {
  const session = await getSession();

  if (!session) {
    // Not authenticated — redirect will be handled by middleware
    window.location.href = "/login";
    return new Response(null, { status: 401 });
  }

  const backendToken = (session as any).backendToken;

  const headers = new Headers(options.headers);
  if (backendToken) {
    headers.set("Authorization", `Bearer ${backendToken}`);
  }
  if (!headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const res = await fetch(url, { ...options, headers });

  // If backend rejects the token, the session might be stale
  if (res.status === 401) {
    // Force re-authentication
    window.location.href = "/login";
  }

  return res;
}

/**
 * Check if the current session user is an admin.
 * For use in client components — prefer middleware for route protection.
 */
export async function checkAdmin(): Promise<boolean> {
  const session = await getSession();
  return (session as any)?.role === "admin" || (session?.user as any)?.role === "admin";
}
