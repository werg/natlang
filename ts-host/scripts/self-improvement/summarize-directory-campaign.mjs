/** Report program repair, restructuring, recovery and transfer without merging them into one score. */
import {readFile,writeFile,readdir} from 'node:fs/promises';
import {join,resolve} from 'node:path';
const root=resolve(process.argv[2]);
const cases=[];
for(const entry of await readdir(root,{withFileTypes:true})){
 if(!entry.isDirectory()||!entry.name.startsWith('directory-'))continue;
 const directory=join(root,entry.name),read=async file=>{try{return JSON.parse(await readFile(join(directory,file),'utf8'));}catch(error){if(error.code==='ENOENT')return null;throw error;}};
 const protocol=await read('protocol.json');if(!protocol)continue;
 const probe=await read('probe.json'),native=await read('native.json'),direct=await read('direct.json'),confirmation=await read('confirm.json');
 const history=native?.state?.history??[];
 const failures=probe?[...probe.baseline.outcomes??[],...probe.reference.outcomes??[]].filter(row=>row.error).map(row=>({caseId:row.caseId,error:row.error})):[];
 cases.push({id:protocol.id,cohort:protocol.cohort,baselineQuality:native?.baseline?.quality??probe?.baseline.quality,referenceTrainQuality:probe?.reference.quality,measuredHeadroom:probe?.headroom,executionFailures:failures,native:native?{disposition:native.disposition,error:native.error,quality:native.validation?.quality,calls:native.validation?.modelCalls,changedPaths:native.sourceDiff.map(change=>change.path),acceptedExperiments:history.filter(step=>step.accepted).length,rejectedExperiments:history.filter(step=>!step.accepted).length,recoveredAfterRejection:history.some((step,i)=>step.accepted&&history.slice(0,i).some(previous=>!previous.accepted)),furtherGain:history.filter(step=>step.accepted).length>1}:null,direct:direct?{accepted:direct.accepted,quality:direct.candidate?.quality,calls:direct.candidate?.modelCalls}:null,confirmation:confirmation?.comparisons??null});
}
let training=null;try{const manifest=JSON.parse(await readFile(join(root,'training-current','manifest.json'),'utf8'));training={rows:manifest.rows,invocations:manifest.cases.reduce((sum,row)=>sum+(row.invocations?.length??0),0),rejectedEditInvocations:manifest.cases.reduce((sum,row)=>sum+(row.failureInvocations?.length??0),0)};}catch(error){if(error.code!=='ENOENT')throw error;}
const report={schema:'natlang.directory-campaign-summary/1',cases,training,interpretation:'Program quality, request efficiency, genuine rejection recovery and transfer are distinct. Timeouts are execution failures, not semantic labels. No weight-learning gain is claimed without the separate frozen checkpoint comparison.'};
await writeFile(join(root,'summary.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
