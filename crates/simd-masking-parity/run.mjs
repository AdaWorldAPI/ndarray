// Node harness for the wasm arms of the masking parity matrix: instantiate the
// .wasm and assert selfcheck() == 0. A nonzero rc names the failing group+op
// (see src/lib.rs, `Code`).
import { readFileSync } from 'node:fs';
const wasmPath = process.argv[2];
if (!wasmPath) {
  console.error('usage: node run.mjs <path-to.wasm>');
  process.exit(2);
}
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const rc = instance.exports.selfcheck();
if (rc === 0) {
  console.log('masking parity: OK (selfcheck rc=0)');
  process.exit(0);
} else {
  console.error(`masking parity: FAIL — selfcheck rc=0x${rc.toString(16)} (see crates/simd-masking-parity/src/lib.rs)`);
  process.exit(1);
}
