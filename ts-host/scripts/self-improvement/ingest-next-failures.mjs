/** Later train-pool failures become another bounded collection, never confirmation-cohort development. */
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const [output,...roots]=process.argv.slice(2);if(!roots.length)throw Error('usage: ingest-next-failures.mjs OUTPUT_DIR TRAIN_FAILURE_ROOT...');
const script=fileURLToPath(new URL('./index-failures.mjs',import.meta.url));
const result=spawnSync(process.execPath,[script,output,...roots],{stdio:'inherit'});process.exit(result.status??1);
