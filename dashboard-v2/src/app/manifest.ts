import type { MetadataRoute } from 'next';
import { APP_NAME, APP_DESCRIPTION, ICONS, APP_NAME_SHORT } from './config';
export default function manifest(): MetadataRoute.Manifest {
        return {
                name: APP_NAME,
                short_name: APP_NAME_SHORT,
                description: APP_DESCRIPTION,
                icons: ICONS,
                start_url: '/',
                theme_color: '#247EB9',
                background_color: '#0B0D0E',
                display: 'standalone',
        };
}
