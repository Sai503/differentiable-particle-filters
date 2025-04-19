import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: '',  // Forces relative paths on build.
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 8000,
    proxy: {
      '/api': 'http://127.0.0.1:5000'
    }
  },
  preview: {
    host: '0.0.0.0',
    port: 8000
  }
});
