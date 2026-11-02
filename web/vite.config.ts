/// <reference types="vitest/config" />
import { createReadStream, existsSync, statSync } from "node:fs";
import { resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv, type Plugin } from "vite";

const MOCK_DIR = fileURLToPath(new URL("./mock", import.meta.url));

/**
 * Serves web/mock at /mock on the dev server only, so the sample data never
 * ships in a production or GitHub Pages build.
 */
function serveMockData(): Plugin {
  return {
    name: "ledgerlens-mock-data",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use("/mock", (req, res, next) => {
        const path = decodeURIComponent((req.url ?? "/").split("?")[0] ?? "/");
        const file = resolve(MOCK_DIR, `.${path}`);
        if (!file.startsWith(MOCK_DIR + sep) || !existsSync(file) || !statSync(file).isFile()) {
          next();
          return;
        }
        res.setHeader("Content-Type", "application/json; charset=utf-8");
        createReadStream(file).pipe(res);
      });
    },
  };
}

// VITE_BASE lets GitHub Pages serve the app under /ledgerlens/.
// VITE_API_PROXY points the dev server at the FastAPI backend.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  return {
    base: env.VITE_BASE || "/",
    plugins: [react(), tailwindcss(), serveMockData()],
    server: {
      proxy: {
        "/api": {
          target: env.VITE_API_PROXY || "http://localhost:8000",
          changeOrigin: true,
        },
      },
    },
    test: {
      environment: "jsdom",
      setupFiles: ["./src/test/setup.ts"],
      css: false,
    },
  };
});
