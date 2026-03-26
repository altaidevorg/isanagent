import path from "node:path";
import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/v1": {
        target: process.env.AGENT_RS_API_ORIGIN ?? "http://127.0.0.1:8080",
        changeOrigin: true,
      },
      "/_mte": {
        target: process.env.AGENT_RS_API_ORIGIN ?? "http://127.0.0.1:8080",
        changeOrigin: true,
      },
    },
  },
});
