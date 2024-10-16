import type { MetadataRoute } from 'next';
import { APP_NAME, APP_DESCRIPTION, ICONS, APP_NAME_SHORT } from './config';
export default function manifest(): MetadataRoute.Manifest {
        return {
                name: APP_NAME,
                short_name: APP_NAME_SHORT,
                description: APP_DESCRIPTION,
                icons: ICONS,
                theme_color: '#000000',
                background_color: '#ffffff',
                display: 'standalone',
        };
}
