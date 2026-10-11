#!/usr/bin/env node
/** Export the runtime's exact prompt scope table for training and serving renderers. */
import { promptPieces } from '../dist/native/system-prompts.js';
process.stdout.write(JSON.stringify({ schema: 'natlang.prompt-pieces/1', pieces: promptPieces() }, null, 2) + '\n');
