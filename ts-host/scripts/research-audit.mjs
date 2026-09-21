import { readFile } from 'node:fs/promises';
import { auditResearchBundle } from '../studio/research/evaluation.mjs';
import { scenarios } from '../studio/research/scenarios.mjs';

const [file, scenario] = process.argv.slice(2);
if (!file || !scenario) {
    console.error(`Usage: node scripts/research-audit.mjs <export.json> <scenario>\nScenarios: ${scenarios.map(row => row.id).join(', ')}`);
    process.exitCode = 2;
} else {
    const report = await auditResearchBundle(JSON.parse(await readFile(file, 'utf8')), scenario);
    console.log(JSON.stringify(report, null, 2));
    if (!report.structural.passed) process.exitCode = 1;
}
