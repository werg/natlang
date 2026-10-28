import { currentSplit } from '../../../ts-host/dist/tune/pipeline.js';

const [root, examples] = process.argv.slice(2);
if (!root || !examples) throw new Error('controller-split requires program and example paths');
const split = currentSplit(root, examples);
console.log(JSON.stringify({ command: 'controller-split', roots: split.length, components: split.receipt.components }));
