import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { cn } from '@/lib/utils';
import { ThemeProvider } from "@/components/theme-provider";
import { Footer } from "@/components/footer";

const inter = Inter({ subsets: ["latin"], variable: "--font-sans" });

export const metadata: Metadata = {
  title: "Durham Rowing - River Dashboard",
  description: "A dashboard to monitor river conditions for rowing in Durham",
  keywords: [
    'Durham',
    'Durham University',
    'Rowing',
    'River',
    'Wear',
    'Weather',
    'Flood',
    'Rain',
    'Water',
    'Data',
  ],
  alternates: {
    canonical: "https://river-level.edjex.dev/",
  },
  authors: [{
    name: "Edward Jex",
    // url: "https://edjex.dev",
  }],
  openGraph: {
    type: "website",
    locale: "en_GB",
    siteName: "Durham Rowing - River Dashboard",
    images: [
      {
        url: "https://river-level.edjex.dev/og-image.png",
        width: 1200,
        height: 630,
        alt: "Durham Rowing - River Dashboard",
      },
    ],
  },

};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={cn(
        "min-h-screen bg-background text-primary-foreground flex flex-col font-sans antialiased",
        inter.variable
      )}>
        <ThemeProvider defaultTheme="system" enableSystem disableTransitionOnChange>
          <div className="flex-1">
            {children}
          </div>
          <Footer />
        </ThemeProvider>
      </body>
    </html>
  );
}
