import react from "@vitejs/plugin-react";
import { copyFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

const rootDir = fileURLToPath(new URL(".", import.meta.url));
const distDir = resolve(rootDir, "dist");

function copyManifest(): Plugin {
  return {
    name: "copy-extension-manifest",
    closeBundle() {
      mkdirSync(distDir, { recursive: true });
      copyFileSync(resolve(rootDir, "manifest.json"), resolve(distDir, "manifest.json"));
    },
  };
}

export default defineConfig({
  plugins: [react(), copyManifest()],
  publicDir: false,
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        index: resolve(rootDir, "index.html"),
        background: resolve(rootDir, "src/background/background.ts"),
      },
      output: {
        entryFileNames(chunkInfo) {
          if (chunkInfo.name === "background") {
            return "background.js";
          }
          return "assets/[name].js";
        },
        chunkFileNames: "assets/[name].js",
        assetFileNames: "assets/[name].[ext]",
      },
    },
  },
});
