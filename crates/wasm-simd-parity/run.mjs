// Node harness for the wasm SIMD parity gate: instantiate the .wasm and assert
// selfcheck() == 0. A nonzero rc names the failing lane+op (see lib.rs codes).
import { readFileSync } from 'node:fs';
const wasmPath = process.argv[2];
if (!wasmPath) {
  console.error('usage: node run.mjs <path-to.wasm>');
  process.exit(2);
}
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const rc = instance.exports.selfcheck();
if (rc === 0) {
  console.log('wasm SIMD parity: OK (selfcheck rc=0)');
  process.exit(0);
} else {
  console.error(`wasm SIMD parity: FAIL — selfcheck rc=${rc} (see crates/wasm-simd-parity/src/lib.rs return codes)`);
  process.exit(1);
}
