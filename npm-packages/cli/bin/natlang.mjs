#!/usr/bin/env node
import { main } from '@natlang/node/cli';
main().then(code => { process.exitCode = code; }, error => {
  process.stderr.write(`natlang: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
});
