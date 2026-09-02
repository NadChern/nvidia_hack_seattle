import { defineConfig } from 'astro/config';

// GitHub Pages project site for the NadChern/nvidia_hack_seattle repository.
// base must match the repository subpath so all assets resolve under it.
export default defineConfig({
  site: 'https://nadchern.github.io/nvidia_hack_seattle/',
  base: '/nvidia_hack_seattle/',
  output: 'static',
  trailingSlash: 'always',
  // Only affects `astro dev` (e.g. sharing via a Cloudflare quick tunnel); no effect on the static build.
  server: {
    allowedHosts: true,
  },
});
