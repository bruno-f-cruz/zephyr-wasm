import { readFileSync } from "node:fs";
import { defineConfig, type Plugin } from "vite";

// Cross-origin isolation lets ONNX Runtime's WebAssembly use threads.
const headers = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

/** ONNX Runtime imports its own loader at run time: serve public/ort/ as it is, since
 * the dev server would otherwise try to transform that import (and fail). */
function rawOrt(): Plugin {
  return {
    name: "raw-ort",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const path = req.url?.split("?")[0] ?? "";
        if (!path.startsWith("/ort/")) return next();
        try {
          const body = readFileSync(`public${path}`);
          res.setHeader("Content-Type", path.endsWith(".wasm") ? "application/wasm" : "text/javascript");
          for (const [k, v] of Object.entries(headers)) res.setHeader(k, v);
          res.end(body);
        } catch {
          next();
        }
      });
    },
  };
}

export default defineConfig({
  base: "./",
  plugins: [rawOrt()],
  server: { headers },
  preview: { headers },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  build: { target: "es2022" },
});
