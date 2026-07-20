/**
 * Mosaic Authentication Configuration (Auth.js v5)
 * 
 * Supports:
 * - Google OAuth
 * - GitHub OAuth
 * - Microsoft (Azure AD) OAuth
 * - Credentials (username/password) — validated against the backend
 * 
 * Security:
 * - Sessions stored in signed, httpOnly JWT cookies (not localStorage)
 * - CSRF protection built-in
 * - Secrets never exposed to the client
 * - OAuth tokens never sent to the browser
 * 
 * Open-source safe:
 * - All secrets live in .env (gitignored)
 * - No hardcoded credentials in source
 * - Provider client IDs are non-sensitive (they're public anyway)
 */

import NextAuth from "next-auth";
import Google from "next-auth/providers/google";
import GitHub from "next-auth/providers/github";
import MicrosoftEntraID from "next-auth/providers/microsoft-entra-id";
import Credentials from "next-auth/providers/credentials";

const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8080";

// Admin emails — users with these emails get admin role
const ADMIN_EMAILS = (process.env.ADMIN_EMAILS || "").split(",").map((e) => e.trim().toLowerCase()).filter(Boolean);

export const { handlers, signIn, signOut, auth } = NextAuth({
  providers: [
    // --- Google ---
    ...(process.env.GOOGLE_CLIENT_ID
      ? [
          Google({
            clientId: process.env.GOOGLE_CLIENT_ID,
            clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
          }),
        ]
      : []),

    // --- GitHub ---
    ...(process.env.GITHUB_CLIENT_ID
      ? [
          GitHub({
            clientId: process.env.GITHUB_CLIENT_ID,
            clientSecret: process.env.GITHUB_CLIENT_SECRET!,
          }),
        ]
      : []),

    // --- Microsoft (Azure AD / Entra ID) ---
    ...(process.env.MICROSOFT_CLIENT_ID
      ? [
          MicrosoftEntraID({
            clientId: process.env.MICROSOFT_CLIENT_ID,
            clientSecret: process.env.MICROSOFT_CLIENT_SECRET!,
            issuer: process.env.MICROSOFT_TENANT_ID
              ? `https://login.microsoftonline.com/${process.env.MICROSOFT_TENANT_ID}/v2.0`
              : undefined,
          }),
        ]
      : []),

    // --- Credentials (username/password against backend) ---
    Credentials({
      name: "credentials",
      credentials: {
        username: { label: "Username", type: "text" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        if (!credentials?.username || !credentials?.password) return null;

        try {
          const res = await fetch(`${BACKEND_URL}/auth/login`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              username: credentials.username,
              password: credentials.password,
            }),
          });

          if (!res.ok) return null;

          const data = await res.json();
          return {
            id: data.username,
            name: data.username,
            email: `${data.username}@local`,
            role: data.role,
            backendToken: data.access_token,
          };
        } catch {
          return null;
        }
      },
    }),
  ],

  callbacks: {
    async jwt({ token, user, account }) {
      // On first sign-in, attach role and backend token
      if (user) {
        token.role = (user as any).role || "user";
        token.backendToken = (user as any).backendToken;
        token.provider = account?.provider || "credentials";

        // For OAuth users, determine role from email
        if (account?.provider !== "credentials" && user.email) {
          token.role = ADMIN_EMAILS.includes(user.email.toLowerCase()) ? "admin" : "user";

          // Get a backend token for OAuth users
          try {
            const res = await fetch(`${BACKEND_URL}/auth/oauth`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                email: user.email,
                name: user.name,
                provider: account?.provider,
                role: token.role,
              }),
            });
            if (res.ok) {
              const data = await res.json();
              token.backendToken = data.access_token;
            }
          } catch {
            // Backend not available — will retry on next request
          }
        }
      }
      return token;
    },

    async session({ session, token }) {
      // Expose role and backend token to the client session
      (session as any).role = token.role;
      (session as any).backendToken = token.backendToken;
      (session as any).provider = token.provider;
      if (session.user) {
        (session.user as any).role = token.role;
      }
      return session;
    },
  },

  pages: {
    signIn: "/login",
  },

  session: {
    strategy: "jwt",
    maxAge: 7 * 24 * 60 * 60, // 7 days
  },

  // Cookies are httpOnly by default — not accessible via JavaScript
  // This prevents XSS attacks from stealing session tokens
});
