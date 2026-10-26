// ONNX Runtime loads its WebAssembly at run time: serve the package's own files.
import { cpSync, mkdirSync, readdirSync } from "node:fs";

const from = "node_modules/onnxruntime-web/dist/";
mkdirSync("public/ort", { recursive: true });
for (const name of readdirSync(from)) {
  if (/^ort-wasm.*\.(wasm|mjs)$/.test(name)) cpSync(from + name, `public/ort/${name}`);
}
