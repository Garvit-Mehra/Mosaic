import type { Metadata } from "next";
import "./globals.css";
import { Roboto } from "next/font/google";
import { SessionProvider } from "next-auth/react";
import SideBarWrapper from "./components/common/SideBarWrapper";

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
    <html lang="en" className={roboto.className}>
      <body className="flex relative h-screen overflow-hidden">
        <SessionProvider>
          <SideBarWrapper />
          <main className="flex-1 h-screen overflow-hidden">{children}</main>
        </SessionProvider>
      </body>
    </html>
  );
}
