import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    allowedHosts: true,
    proxy: {
      "/api": "http://127.0.0.1:8765"
    }
  },
  preview: {
    port: 5175,
    allowedHosts: true,
    proxy: {
      "/api": "http://127.0.0.1:8765"
    }
  }
});
