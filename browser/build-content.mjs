/**
 * Build the content script with esbuild, wrapping the output so that
 * browser.tabs.executeScript captures the return value of extract().
 */
import { build } from "esbuild";
import { writeFileSync, mkdirSync } from "fs";

const result = await build({
  entryPoints: ["src/content/extract_text.ts"],
  bundle: true,
  format: "esm",
  target: "es2020",
  write: false,
});

// Wrap in an IIFE that returns the result of extract()
// executeScript captures the value of the last evaluated expression
const code = result.outputFiles[0].text;
// The ESM output ends with `extract();` — replace with `return extract();`
const wrapped = `(() => {\n${code.replace(/extract\(\);\s*$/, "return extract();\n")}\n})();`;

mkdirSync("dist/content", { recursive: true });
writeFileSync("dist/content/extract_text.js", wrapped);
console.log(`  dist/content/extract_text.js  ${(wrapped.length / 1024).toFixed(1)}kb`);
