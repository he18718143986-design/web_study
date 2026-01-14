// frontend/vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/v1': {
        target: 'https://reimagined-potato-7v6wvgp5pr5qhxj4q-8000.app.github.dev',
        changeOrigin: true,
        secure: false
      }
    }
  }
})
