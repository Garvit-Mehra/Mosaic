import type { Metadata } from "next";
import "./globals.css";
import { Roboto } from "next/font/google";
import { SessionProvider } from "next-auth/react";
import SideBarWrapper from "./components/common/SideBarWrapper";
import KeyboardShortcuts from "./components/common/KeyboardShortcuts";
import { ErrorBoundary } from "./components/common/ErrorBoundary";
import { ThemeProvider } from "@/src/lib/theme";

const roboto = Roboto({
  weight: ["400", "500"],
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Mosaic",
  description: "A modular multi-agent AI assistant",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={roboto.className} suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              try {
                const theme = localStorage.getItem('mosaic_theme') || 'dark';
                document.documentElement.setAttribute('data-theme', theme);
              } catch (e) {}
            `,
          }}
        />
      </head>
      <body className="flex relative h-screen overflow-hidden">
        <KeyboardShortcuts />
        <ErrorBoundary>
          <ThemeProvider>
            <SessionProvider refetchInterval={0} refetchOnWindowFocus={false}>
              <SideBarWrapper />
              <main className="flex-1 h-screen overflow-hidden">{children}</main>
            </SessionProvider>
          </ThemeProvider>
        </ErrorBoundary>
      </body>
    </html>
  );
}
