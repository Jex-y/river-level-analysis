import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Footer } from '@/components/footer';
import { ThemeProvider } from '@/components/theme-provider';
import { cn } from '@/lib/utils';

const inter = Inter({ subsets: ['latin'], variable: '--font-sans' });

export const metadata: Metadata = {
        title: 'Durham Rowing - River Dashboard',
        description: 'A dashboard to monitor river conditions for rowing in Durham',
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
                canonical: 'https://river.edjex.dev/',
        },
        authors: [
                {
                        name: 'Edward Jex',
                        // url: "https://edjex.dev",
                },
        ],
        openGraph: {
                type: 'website',
                locale: 'en_GB',
                siteName: 'Durham Rowing - River Dashboard',
        },
        icons: {
                shortcut: { type: 'image/x-icon', url: '/favicon.ico' },
                icon: [
                        {
                                url: '/favicon.ico',
                                type: 'image/x-icon',
                        },
                        {
                                url: '/favicon.svg',
                                type: 'image/svg+xml',
                        },
                        ...[16, 32, 64, 96, 128, 192, 256, 512].map(
                                (size) =>
                                        ({
                                                url: `/icon-${size}.png`,
                                                type: 'image/png',
                                                sizes: `${size}x${size}`,
                                        }) as const
                        ),
                ],
                apple: [
                        {
                                url: '/apple-touch-icon.png',
                                sizes: '180x180',
                                type: 'image/png',
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
                        <body
                                className={cn(
                                        'min-h-screen bg-background text-primary-foreground flex flex-col font-sans antialiased',
                                        inter.variable
                                )}
                        >
                                <ThemeProvider
                                        defaultTheme="system"
                                        enableSystem
                                        disableTransitionOnChange
                                >
                                        <div className="flex-1">{children}</div>
                                        <Footer />
                                </ThemeProvider>
                        </body>
                </html>
        );
}
