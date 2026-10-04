#!/usr/bin/env node
/** Provider-free episode integrity, source-role and independent reference audit. */
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { validateEpisode } from '../../dist/skills/episode.js';
import { exactObjectiveBounds } from '../../dist/skills/objective.js';
const hash = value => createHash('sha256').update(value).digest('hex');
const canonical = value => Array.isArray(value) ? value.map(canonical) : value && typeof value === 'object'
  ? Object.fromEntries(Object.entries(value).sort(([a],[b]) => a.localeCompare(b)).map(([key,item]) => [key,canonical(item)])) : value;
const files = [], outAt = process.argv.indexOf('--out');
for (let i=2;i<process.argv.length;i++) { if (process.argv[i] === '--out') { i++; continue; } files.push(process.argv[i]); }
if (!files.length) throw Error('Usage: audit-episodes.mjs PACKET... [--out REPORT.json]');
const groups = new Map(), inputs = new Map(), ids = new Set(), errors = [], packets = [];
let episodes=0,cases=0,references=0;
for (const file of files) {
  const bytes = await readFile(file), rows=bytes.toString().split('\n').filter(x=>x.trim()).map(JSON.parse);
  packets.push({path:file,sha256:hash(bytes),episodes:rows.length});
  for (const episode of rows) {
    episodes++; if(ids.has(episode.id)) errors.push({code:'duplicate_episode',episode:episode.id});ids.add(episode.id);
    const schemaErrors=validateEpisode(episode);
    errors.push(...schemaErrors.map(error=>({episode:episode.id,...error})));
    if(schemaErrors.length) continue;
    for (const [role,list,target,metric] of [ ['support',episode.support.cases,episode.target,episode.provenance?.metric],
      ['query',episode.query.cases,episode.target,episode.provenance?.metric],
      ['transfer',episode.transfer?.cases??[],episode.transfer?.target,episode.provenance?.transfer_metric] ]) {
      for (const row of list) {
        cases++; const boundary=episode.split+'/'+role;
        const membership={episode:episode.id,role,split:episode.split,case:row.id};
        if(groups.has(row.group) && groups.get(row.group).boundary!==boundary)
          errors.push({code:'group_boundary_collision',group:row.group,first:groups.get(row.group).membership,second:membership});
        else groups.set(row.group,{boundary,membership});
        const task=metric?.schema==='natlang.skill-objective/1' ? metric.kind : target?.files?.[target.entry];
        const signature=hash(JSON.stringify(canonical({task,args:row.args,folder:row.folder})));
        const previous=inputs.get(signature);
        if(previous && previous.boundary!==boundary)
          errors.push({code:'input_alias_boundary_collision',input_sha256:signature,first:previous.membership,second:membership});
        else inputs.set(signature,{boundary,membership});
        if(metric?.schema==='natlang.skill-objective/1') {
          try {
            const actual=exactObjectiveBounds(metric.kind,typeof row.args[0]==='string'?JSON.parse(row.args[0]):row.args[0]);
            if(JSON.stringify(canonical(actual))!==JSON.stringify(canonical(row.expected))) throw Error('reference bound mismatch');
            references++;
          } catch(error) {errors.push({code:'independent_reference_error',episode:episode.id,case:row.id,error:String(error)});}
        }
      }
    }
  }
}
const report={schema:'natlang.skill-episode-audit/1',provider_calls:0,packets,episodes,cases,groups:groups.size,
  canonical_inputs:inputs.size,independent_objective_references_checked:references,errors,passed:errors.length===0};
if(outAt>=0) await writeFile(process.argv[outAt+1],JSON.stringify(report,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({...report,errors:errors.slice(0,10)},null,2));
if(errors.length) process.exitCode=1;
