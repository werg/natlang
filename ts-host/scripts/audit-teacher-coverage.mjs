#!/usr/bin/env node
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { auditCoverage } from './teacher-coverage.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const args = process.argv.slice(2), value = (flag, fallback) => {
  const index = args.indexOf(flag); return resolve(index < 0 ? fallback : args[index + 1]);
};
const report = await auditCoverage(root,
  value('--config', resolve(root, 'training/teacher_coverage.json')),
  value('--studio-cases', resolve(root, 'data/teacher/studio-cases-v1.jsonl')),
  value('--program-ir', resolve(root, 'data/teacher/coverage-selection-s909.ir.jsonl')));
console.log(JSON.stringify(report, null, 2));
if (!report.ready) process.exitCode = 2;
