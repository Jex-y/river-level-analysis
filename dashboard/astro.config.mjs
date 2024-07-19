import react from '@astrojs/react';
import { defineConfig } from 'astro/config';

import tailwind from '@astrojs/tailwind';

// https://astro.build/config
export default defineConfig({
	output: 'hybrid',
	integrations: [
		react({ experimentalReactChildren: true }),
		tailwind({ applyBaseStyles: false }),
	],
});
