#!/usr/bin/env node
import { parseArgs } from 'node:util';
import { verifyCandidateSelectionPacket } from './lib/source-quality-clearance-sidecar.mjs';

const { values } = parseArgs({ options: {
  root: { type: 'string' }, 'manifest-sha256': { type: 'string' },
} });
if (!values.root || !/^[a-f0-9]{64}$/.test(values['manifest-sha256'] ?? ''))
  throw new Error('usage: node scripts/verify_candidate_selection_packet.mjs --root DIR --manifest-sha256 SHA256');
const verified = await verifyCandidateSelectionPacket(values.root, values['manifest-sha256']);
process.stdout.write(JSON.stringify(verified) + '\n');
