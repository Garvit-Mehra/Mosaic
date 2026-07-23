/**
 * Mosaic Auth Client Utilities
 * 
 * Provides authFetch() that accepts a pre-fetched session token,
 * avoiding repeated /api/auth/session calls.
 */

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL;

export interface User {
  name?: string | null;
  email?: string | null;
  image?: string | null;
  role?: "admin" | "user";
}

/**
 * Make an authenticated fetch to the backend.
 * Pass the backendToken from useSession() — does NOT call getSession().
 * If no token is provided, the request is made without auth (backend will reject if needed).
 */
export async function authFetch(
  url: string,
  options: RequestInit = {},
  backendToken?: string | null,
): Promise<Response> {
  const headers = new Headers(options.headers);

  if (backendToken) {
    headers.set("Authorization", `Bearer ${backendToken}`);
  }
  if (!headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const res = await fetch(url, { ...options, headers });

  if (res.status === 401) {
    window.location.href = "/login";
  }

  return res;
}
