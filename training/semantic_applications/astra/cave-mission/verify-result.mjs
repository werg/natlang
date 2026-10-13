/** External exact verifier: replay a supplied policy; never select its next action. */
const canonical = v => Array.isArray(v) ? v.map(canonical) : v && typeof v === 'object' ? Object.fromEntries(Object.keys(v).sort().map(k => [k,canonical(v[k])])) : v;
const eq = (a,b) => JSON.stringify(canonical(a)) === JSON.stringify(canonical(b));
const setEq = (a,b) => Array.isArray(a) && Array.isArray(b) && a.length === new Set(a).size && b.length === new Set(b).size && eq([...a].sort(),[...b].sort());
export function verifyCaveResult(report, oracle, documents, request) {
  const issues=[]; const qualityObservations=[]; const check=(v,m)=>{if(!v)issues.push(m);};
  try {
    check(report.caseId===request.caseId,'caseId');
    const docs=new Map(documents.map(d=>[d.id,d]));
    const walk=(value,path='report')=>{
      if(!value || typeof value!=='object')return;
      if('sourceId' in value && 'quote' in value){const d=docs.get(value.sourceId);check(d && d.path===value.path && typeof value.quote==='string' && value.quote.length>0 && d.text.includes(value.quote),`${path}: witness`);}
      for(const [k,v] of Object.entries(value))walk(v,`${path}/${k}`);
    };walk(report);
    check(setEq(report.sourceCoverage.map(x=>x.sourceId),documents.map(x=>x.id)),'coverage documents');
    for(const d of documents){const ids=[...d.text.matchAll(/^\[([A-Z]\d+)\]/gm)].map(x=>x[1]);const c=report.sourceCoverage.find(x=>x.sourceId===d.id);check(c&&setEq(c.clauseIds,ids),`coverage ${d.id}`);}
    const worlds=report.worlds.worlds;check(worlds.length===oracle.worlds.length,'world count');
    check(setEq(worlds.map(w=>w.worldId),worlds.map(w=>w.worldId)),'unique world IDs');
    check(report.worlds.rejected.length===0 && report.worlds.unresolved.length===0,'unexpected rejected/unresolved world');
    const semantics=new Map(); const signs=[];
    for(const w of worlds){
      const t=w.frame.transform;const sign=t.uVector.x;signs.push(sign);const gold=oracle.worlds.find(x=>x.uToGlobalXSign===sign);
      check(!!gold,`world ${w.worldId}: transform sign`);if(!gold)continue;
      check(eq(t.origin,{x:gold.origin[0],y:gold.origin[1]})&&eq(t.uVector,{x:sign,y:0})&&eq(t.vVector,{x:0,y:1}),`world ${w.worldId}: affine transform`);
      const bindings={origin:['J',0,1],A:[gold.A,sign,1],B:[gold.B,-sign,1],V:['V',2*sign,1],X:['X',-2*sign,1]};
      check(setEq(w.frame.bindings.map(x=>x.localId),Object.keys(bindings)),`world ${w.worldId}: binding domain`);
      for(const b of w.frame.bindings){const g=bindings[b.localId];check(g&&b.globalId===g[0]&&eq(b.position,{x:g[1],y:g[2]}),`world ${w.worldId}: binding ${b.localId}`);}
      const edges=[['E','J','corridor',1],['J','E','corridor',1],['J','EW','water-gate',1],['J','WW','water-gate',1],[gold.A,'V','tunnel',1],['V',gold.A,'tunnel',1],[gold.B,'X','tunnel',1],['X',gold.B,'tunnel',1],['V','J','chimney',2]];
      const key=e=>JSON.stringify(e);check(setEq(w.passages.map(p=>key([p.from,p.to,p.kind,p.cost])),edges.map(key)),`world ${w.worldId}: directed passages`);
      check(w.sampleLocation==='V'&&w.correctRigSide===gold.rigSide,`world ${w.worldId}: sample/rig`);
      check(setEq(w.facts.map(x=>x.id),documents.flatMap(d=>[...d.text.matchAll(/^\[([A-Z]\d+)\]/gm)].map(x=>x[1]))),`world ${w.worldId}: fact coverage`);
      semantics.set(w.worldId,{...gold,edges});
    }
    check(setEq(signs,oracle.worlds.map(x=>x.uToGlobalXSign)),'frame alternatives complete');
    const bindings=report.worlds.landmarks.fixedBindings;
    check(bindings.some(b=>b.localId==='origin'&&b.globalId==='J'),'shared origin anchor');
    check(!bindings.some(b=>['A','B'].includes(b.localId)&&['EW','WW'].includes(b.globalId)),'unsupported fixed landing');
    check(!report.worlds.landmarks.decisions.some(d=>d.relation==='same'&&((['A','B'].includes(d.leftId)&&['EW','WW'].includes(d.rightId))||(['V','X'].includes(d.leftId)&&['V','X'].includes(d.rightId)&&d.leftId!==d.rightId))),'unsupported landmark identity');
    const policy=report.policy;const nodes=new Map(policy.nodes.map(n=>[n.nodeId,n]));
    check(nodes.size===policy.nodes.length,'duplicate policy node');check(setEq(policy.nodes.filter(n=>n.action).map(n=>n.action.actionId),policy.nodes.filter(n=>n.action).map(n=>n.action.actionId)),'duplicate action ID');check(nodes.has(policy.rootNodeId),'root node');check(policy.blockedReasons.length===0,'blocked policy');
    check(setEq(report.replays.map(x=>x.worldId),worlds.map(x=>x.worldId)),'replay world coverage');
    const used=new Set();const totals=[];
    for(const world of worlds){
      const g=semantics.get(world.worldId);if(!g)continue;
      let robot={location:request.start,energy:request.initialEnergy,sampleHeld:false,rope:'unused',rigSide:null};
      let current=policy.rootNodeId,belief=worlds.map(w=>w.worldId),observations=0,success=false;
      const visited=[],actions=[];
      while(current!==null){
        if(visited.includes(current)){check(false,`cycle ${world.worldId}`);break;}
        const n=nodes.get(current);if(!n){check(false,`missing node ${current}`);break;}
        visited.push(current);used.add(current);
        if(n.terminal){check(n.action===null&&n.outcomes.length===0,'terminal shape');success=robot.location===request.start&&robot.sampleHeld&&robot.energy>=0;check(success,`goal ${world.worldId}`);break;}
        const a=n.action;if(!a){check(false,'nonterminal action absent');break;}
        actions.push(a.actionId);check(a.from===robot.location,`action source ${a.actionId}`);check(a.witnesses.length>0,`action witness ${a.actionId}`);
        let cost=null,answer=null;
        if(a.kind==='move'){
          const edge=g.edges.find(e=>e[0]===robot.location&&e[1]===a.to);check(!!edge,`unsupported move ${a.actionId}`);
          if(edge){cost=edge[3];check(edge[2]!=='tunnel'||!robot.sampleHeld,`full carrier tunnel ${a.actionId}`);check(edge[2]!=='chimney'||(robot.rope==='rigged'&&robot.rigSide===g.rigSide),`chimney anchor ${a.actionId}`);}
          check(a.rigSide===null&&a.sensorId===null,'movement fields');robot.location=a.to;
        }else if(a.kind==='rig'){
          cost=1;check(robot.location==='J'&&robot.rope==='unused'&&['east','west'].includes(a.rigSide),'rig precondition');check(a.to===null&&a.sensorId===null,'rig fields');robot.rope='rigged';robot.rigSide=a.rigSide;
        }else if(a.kind==='pickup'){
          cost=1;check(robot.location==='V'&&!robot.sampleHeld,'pickup precondition');check(a.to===null&&a.rigSide===null&&a.sensorId===null,'pickup fields');robot.sampleHeld=true;
        }else if(a.kind==='observe'){
          cost=1;observations++;check(robot.location==='J'&&['ECHO','GLOW'].includes(a.sensorId),'sensor precondition');check(a.to===null&&a.rigSide===null,'observe fields');
          answer=a.sensorId==='ECHO'?g.echo:g.glow;
        }else check(false,'action kind');
        check(a.cost===cost,`cost ${a.actionId}`);robot.energy-=a.cost;check(robot.energy>=0,`energy ${a.actionId}`);
        const expected=new Map();
        for(const wid of belief){const wg=semantics.get(wid);const ans=a.kind==='observe'?(a.sensorId==='ECHO'?wg?.echo:wg?.glow):null;const key=JSON.stringify(ans);expected.set(key,[...(expected.get(key)||[]),wid]);}
        check(n.outcomes.length===expected.size,`outcome count ${n.nodeId}`);
        check(setEq(n.outcomes.map(o=>JSON.stringify(o.answer)),[...expected.keys()]),`answer domain ${n.nodeId}`);
        for(const o of n.outcomes)check(setEq(o.worldIds,expected.get(JSON.stringify(o.answer))||[]),`answer world partition ${n.nodeId}`);
        const candidates=n.outcomes.filter(o=>o.answer===answer&&o.worldIds.includes(world.worldId));
        if(candidates.length!==1){check(false,`actual branch ${n.nodeId}`);break;}
        belief=candidates[0].worldIds;current=candidates[0].nextNodeId;
      }
      const replay=report.replays.find(x=>x.worldId===world.worldId);
      check(replay&&eq(replay.nodeIds,visited)&&eq(replay.actionIds,actions),`replay chronology ${world.worldId}`);
      check(replay&&eq(replay.terminal,robot)&&replay.totalEnergy===request.initialEnergy-robot.energy&&replay.observations===observations&&replay.success===success,`replay state ${world.worldId}`);
      check(replay&&replay.findings.every(x=>x.kind==='defended'),'replay unresolved finding');
      totals.push({energy:request.initialEnergy-robot.energy,observations});
    }
    if(used.size!==nodes.size)qualityObservations.push({kind:'unused-policy-nodes',nodeIds:[...nodes.keys()].filter(id=>!used.has(id))});
    check(policy.worstCaseEnergy===Math.max(...totals.map(x=>x.energy))&&policy.worstCaseEnergy===oracle.minimum_worst_case_energy,'minimax energy');
    check(policy.worstCaseObservations===Math.max(...totals.map(x=>x.observations))&&policy.worstCaseObservations===oracle.minimum_observations_at_optimum,'minimax observations');
    check(report.findings.every(x=>x.kind==='defended'),'unresolved/defect finding');
  } catch(e){issues.push(`malformed report: ${e.message}`);}
  return {ok:issues.length===0,issues,qualityObservations,semantic_review_required:true,scope:'Exact source-world structure and all-world supplied-policy replay; source quotation entailment and prose remain independently reviewed.'};
}
export function verifyCandidate({value,oracle,inputs,fixture}) {
  try {
    const manifest=JSON.parse(fixture['manifest.json']);const request=inputs.request??inputs;
    const documents=manifest.documents.map(d=>({...d,text:fixture[d.path]}));
    if(manifest.caseId!==request.caseId)return {ok:false,issues:['request/fixture case mismatch']};
    return verifyCaveResult(value,oracle,documents,request);
  }catch(e){return {ok:false,issues:[`invalid verifier input: ${e.message}`]};}
}
