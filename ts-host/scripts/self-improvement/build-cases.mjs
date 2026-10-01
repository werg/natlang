import { readFile,writeFile,mkdir } from 'node:fs/promises';
import { join } from 'node:path';
const [targets,output]=process.argv.slice(2);if(!output)throw Error('usage: build-cases.mjs TARGETS_DIR OUTPUT_DIR');
await mkdir(output,{recursive:true});
const rows=(await readFile(join(targets,'targets.jsonl'),'utf8')).trim().split('\n').filter(Boolean).map(JSON.parse);
const counts={}, selected=[];
for(const row of rows){if((counts[row.family]??0)>=10 || selected.length>=120)continue;counts[row.family]=(counts[row.family]??0)+1;selected.push({...row,version:'natlang.improvement-case/1',mechanism:row.incidents.some(i=>i.recovered)?'recovered-failure':'program-repair',policy:{maxExperiments:3,maxPopulation:3,mode:'instruction',strategy:'adaptive',goal:row.goal,allowedFiles:Object.keys(row.files)}});}
await writeFile(join(output,'cases.jsonl'),selected.map(JSON.stringify).join('\n')+'\n');
await writeFile(join(output,'manifest.json'),JSON.stringify({version:'natlang.improvement-cases/1',count:selected.length,families:counts,coverageMissing:Math.max(0,120-selected.length),allocation:{maxIncidents:120,maxClusterAttempts:3,maxCandidates:3,maxImproverRequests:104,maxTargetCaseExecutions:40},terminalDisposition:'collection-pending'},null,2));
console.log(JSON.stringify({cases:selected.length,families:counts}));
