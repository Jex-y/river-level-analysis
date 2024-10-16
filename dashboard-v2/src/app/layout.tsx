import type { Metadata } from 'next';
import {
        APP_NAME,
        APP_DESCRIPTION,
        KEYWORDS,
        ICON_ICO,
        ICON_SVG,
        ICONS_PNG,
        APPLE_ICON,
} from './config';
import { Inter } from 'next/font/google';
import './globals.css';
import { Footer } from '@/components/footer';
import { ThemeProvider } from '@/components/theme-provider';
import { cn } from '@/lib/utils';

const inter = Inter({ subsets: ['latin'], variable: '--font-sans' });
const map_icon = ({ src, type, sizes }: typeof ICON_SVG) => ({
        url: src,
        type: type,
        sizes: sizes,
});

export const metadata: Metadata = {
        title: APP_NAME,
        description: APP_DESCRIPTION,
        keywords: KEYWORDS,
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
                shortcut: {
                        url: ICON_ICO.src,
                        type: ICON_ICO.type,
                },
                icon: [ICON_SVG, ...ICONS_PNG].map(map_icon),
                apple: [APPLE_ICON, ...ICONS_PNG].map(map_icon),
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
