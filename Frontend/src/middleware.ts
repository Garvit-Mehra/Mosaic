import { auth } from "@/src/auth";
import { NextResponse } from "next/server";

export default auth((req) => {
  const { pathname } = req.nextUrl;

  // Public paths — no auth needed
  const publicPaths = ["/login", "/api/auth"];
  const isPublicPath = publicPaths.some((p) => pathname.startsWith(p));

  // If user is logged in and tries to go to login, redirect to home
  if (isPublicPath && pathname.startsWith("/login") && req.auth) {
    return NextResponse.redirect(new URL("/", req.nextUrl.origin));
  }

  if (isPublicPath) {
    return NextResponse.next();
  }

  // Not authenticated — redirect to login
  if (!req.auth) {
    const loginUrl = new URL("/login", req.nextUrl.origin);
    loginUrl.searchParams.set("callbackUrl", pathname);
    return NextResponse.redirect(loginUrl);
  }

  // Admin-only paths
  const adminPaths = ["/admin"];
  if (adminPaths.some((p) => pathname.startsWith(p))) {
    if ((req.auth as any)?.role !== "admin" && (req.auth.user as any)?.role !== "admin") {
      return NextResponse.redirect(new URL("/", req.nextUrl.origin));
    }
  }

  return NextResponse.next();
});

export const config = {
  matcher: [
    // Protect everything except static files and API auth routes
    "/((?!_next/static|_next/image|favicon.ico|api/auth).*)",
  ],
};
