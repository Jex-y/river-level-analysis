import react from '@astrojs/react';
import tailwind from '@astrojs/tailwind';
import { defineConfig } from 'astro/config';


// https://astro.build/config
export default defineConfig({
  output: 'static',
  site: 'https://river-level.edjex.dev',
  integrations: [react({
    experimentalReactChildren: true
  }), tailwind({
    applyBaseStyles: false
  })],
});