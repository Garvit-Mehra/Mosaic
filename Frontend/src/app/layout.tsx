import type { Metadata } from "next";
import "./globals.css";
import { Roboto } from "next/font/google";
import SideBar from "./components/common/SideBar";

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
        <SideBar />
        <main className="flex-1 h-screen overflow-hidden">{children}</main>
      </body>
    </html>
  );
}
