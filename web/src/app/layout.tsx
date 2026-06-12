import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Nav } from "@/components/nav";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Lead Scoring & Buying Groups — B2B account targeting demo",
  description:
    "Portfolio demo: a propensity model and buying-group completeness framework that together decide which B2B accounts to work and what to fix at each one.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col">
        <Nav />
        <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8">{children}</main>
        <footer className="border-t border-border-subtle bg-white">
          <div className="mx-auto max-w-6xl space-y-2 px-4 py-6 text-xs text-muted">
            <p>Independent portfolio project by Philip Felix.</p>
            <p>
              Every number is computed from a seeded synthetic CRM dataset (100K accounts, 718K contacts, 50K
              opportunities) where the win-probability signals are planted by design — so results validate that the
              method recovers known signals, and absolute lifts should not be read as real-world findings. The full
              honest framing is on the Methodology page.{" "}
              <a href="https://github.com/pfelix828/lead-scoring" className="underline decoration-dotted underline-offset-2 hover:text-foreground">
                Code on GitHub
              </a>
              .
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
