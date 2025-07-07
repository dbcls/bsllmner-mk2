import react from "@vitejs/plugin-react-swc"
import path from "path"
import { defineConfig } from "vite"

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  root: "./src",
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
  build: {
    outDir: "../dist",
  },
  server: {
    host: process.env.BSLLMNER2_FRONT_HOST || "0.0.0.0",
    port: parseInt(process.env.BSLLMNER2_FRONT_PORT || "3000"),
    proxy: {
      "/api": {
        target: process.env.BSLLMNER2_API_INTERNAL_URL || "http://bsllmner-mk2-api:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
      "/ollama": {
        target: process.env.BSLLMNER2_OLLAMA_URL || "http://bsllmner-mk2-ollama:11434",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/ollama/, ""),
      },
    },
  },
  preview: {
    host: process.env.BSLLMNER2_FRONT_HOST || "0.0.0.0",
    port: parseInt(process.env.BSLLMNER2_FRONT_PORT || "3000"),
  },
  define: {
    __APP_VERSION__: JSON.stringify(process.env.npm_package_version || "0.0.0"),
    BSLLMNER2_FRONT_BASE: JSON.stringify(process.env.BSLLMNER2_FRONT_BASE || "/"),
    BSLLMNER2_API_URL: JSON.stringify(
      process.env.BSLLMNER2_FRONT_EXTERNAL_URL ?
        `${process.env.BSLLMNER2_FRONT_EXTERNAL_URL}/api` :
        "http://localhost:3000/api",
    ),
    BSLLMNER2_OLLAMA_URL: JSON.stringify(
      process.env.BSLLMNER2_FRONT_EXTERNAL_URL ?
        `${process.env.BSLLMNER2_FRONT_EXTERNAL_URL}/ollama` :
        "http://localhost:3000/ollama",
    ),
  },
  base: process.env.BSLLMNER2_FRONT_BASE || "/",
})
