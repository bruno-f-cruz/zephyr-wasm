import { defineConfig } from "vite";

// Cross-origin isolation lets ONNX Runtime's WebAssembly use threads.
const headers = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  base: "./",
  server: { headers },
  preview: { headers },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  build: { target: "es2022" },
});
