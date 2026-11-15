// Runs a Rust test binary built for wasm32-wasip1 under node's WASI, single
// threaded (wasm32-wasip1 has no threads). Exit code: the test binary's.
//
//   node scripts/wasi-test-run.mjs target/wasm32-wasip1/release/deps/add_pm-*.wasm
import { readFile } from 'node:fs/promises';
import { WASI } from 'node:wasi';

const wasi = new WASI({ version: 'preview1', args: ['test', '--test-threads=1'], env: {}, preopens: {} });
const module = await WebAssembly.compile(await readFile(process.argv[2]));
const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
process.exitCode = wasi.start(instance);
