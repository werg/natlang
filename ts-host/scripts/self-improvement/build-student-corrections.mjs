import {readFile,writeFile,readdir} from 'node:fs/promises';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
const [directory,output]=process.argv.slice(2);const cases=[];
for(const file of await readdir(directory)){
 if(file==='summary.json'||!file.endsWith('.json'))continue;
 const text=await readFile(join(directory,file),'utf8'),row=JSON.parse(text);if(row.accepted||!row.caseDefinition)continue;
 const last=row.exchanges.at(-1)?.wireResponse?.choices?.[0]?.message;
 const next=structuredClone(row.caseDefinition);next.id='student-correction-'+row.id;
 next.policy.outerLesson='Previous student improver failed: '+row.error+'. Last action: '+JSON.stringify(last).slice(0,700)+'. Run the real orchestration and initialize measured baseline; do not fabricate source identities, measurements or completed state.';
 next.studentFailure={path:join(directory,file),sha256:createHash('sha256').update(text).digest('hex'),checkpoint:row.checkpoint};
 next.incidents.push({id:'student-failure:'+next.studentFailure.sha256,cluster:next.incidents[0].cluster,reason:row.error,provenance:'actual-round-1-student-action'});
 cases.push(next);
}
await writeFile(output,cases.map(JSON.stringify).join('\n')+'\n');console.log(JSON.stringify({corrections:cases.length}));
