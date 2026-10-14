import fs from 'node:fs';
const p=JSON.parse(fs.readFileSync('plans/semantic_stack/ASTRA_ECOLOGY_ENTITY_SURFACES_V14_PROPOSAL.json','utf8'));
const idiom=p.typed_evidence_handoff.stage1.direct_await_idiom;
const snippet=idiom.slice(idiom.indexOf('const lexicalCharacter'),idiom.indexOf('const assertion:'));
const check=new Function('record','content',snippet);
const rows=[['plural source copied','Foxes reduce hares.','Foxes','foxes',true],['singular prefix rejected','Foxes reduce hares.','Fox','fox',false],['inflected key rejected','Foxes reduce hares.','Foxes','fox',false],['multiword source copied','Canopy reduces ground light.','ground light','ground light',true],['empty span rejected','Foxes reduce hares.','','',false],['absent span rejected','Foxes reduce hares.','wolves','wolves',false],['casefold only','Hares reduce grass.','Hares','hares',true],['later complete occurrence','Foxes differ from Fox.','Fox','fox',true]];
const controls=rows.map(([name,text,sourceForm,key,expected])=>{let actual=true;try{check({text},{entitySurfaces:[{sourceForm,key}]});}catch{actual=false;}return {name,expected,actual,passed:actual===expected};});
console.log(JSON.stringify({passed:controls.every(x=>x.passed),controls},null,2));if(controls.some(x=>!x.passed))process.exitCode=1;
