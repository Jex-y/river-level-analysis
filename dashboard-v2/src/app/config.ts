export const APP_NAME = 'Durham Rowing - River Dashboard';
export const APP_NAME_SHORT = 'River Level';
export const APP_DESCRIPTION =
        'Monitor rowing conditions on the river Wear in Durham' as const;
export const KEYWORDS = [
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
];

export const ICON_ICO = {
        src: '/icon.ico',
        type: 'image/x-icon',
        sizes: '48x48',
};
export const ICON_SVG = {
        src: '/icon.svg',
        type: 'image/svg+xml',
        sizes: 'any',
};

export const ICONS_PNG = [16, 32, 64, 96, 128, 192, 256, 512].map((size) => ({
        src: `/icon-${size}.png`,
        type: 'image/png',
        sizes: `${size}x${size}`,
}));

export const APPLE_ICON = {
        src: 'apple-icon.png',
        type: 'image/png',
        sizes: '180x180',
};

export const ICONS = [ICON_ICO, ICON_SVG, ...ICONS_PNG];
