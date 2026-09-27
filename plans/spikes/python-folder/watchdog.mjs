import { workerData } from 'node:worker_threads';
const flag = new Int32Array(workerData.buffer);
setTimeout(() => { Atomics.store(flag, 0, 2); }, workerData.ms);   // 2 = SIGINT for Pyodide's interrupt buffer
