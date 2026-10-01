import type {ImprovementPolicy,RewriteRequest,TrainingEvidence,SourceFile,ExperimentOutcome} from '../types';

/** Pass the actual training evidence through unchanged; the model chooses only its hypothesis. */
export function request(policy:ImprovementPolicy,hypothesis:string,evidence:unknown[],sourceFiles:SourceFile[],history:ExperimentOutcome[]=[],lastExperiment?:unknown):RewriteRequest {
  return {history,...(lastExperiment?{lastExperiment}:{}),brief:brief(policy,evidence as TrainingEvidence[],sourceFiles,hypothesis),objective:policy.objective??'quality',goal:policy.goal,mode:policy.mode,hypothesis,sourceFiles,evidence,allowedFiles:policy.allowedFiles};
}

/** A small first read; complete source and observations remain available in the typed context. */
export function brief(policy:ImprovementPolicy,evidence:TrainingEvidence[],sourceFiles:SourceFile[],hypothesis:string=''):string {
  const lines=['Goal: '+policy.goal,'Mode: '+policy.mode+'; objective: '+(policy.objective??'quality'),'Allowed source edits: '+policy.allowedFiles.join(', '),'Execution output files belong to the target runtime folder; implement their behavior in source, not in this draft.'];
  if(hypothesis)lines.push('Hypothesis: '+hypothesis);
  for(const file of sourceFiles)lines.push('Source '+file.path+':\n'+file.text.slice(0,1200)+(file.text.length>1200?'\n[clipped; full text in sourceFiles]':''));
  for(const row of evidence){
    lines.push('Training '+(row.caseId??'case')+': '+(row.passed?'passed':'failed')+'; model requests '+(row.modelCalls??'unknown')+(row.failureKind?'; '+row.failureKind:'')+(row.error?'; '+row.error.slice(0,400):''));
    if(!row.passed)lines.push('Observed: '+JSON.stringify({args:row.args,value:row.value,expected:row.expected}).slice(0,700));
    if(!row.passed&&row.expectedFiles){
      const paths=[...new Set([...Object.keys(row.expectedFiles),...Object.keys(row.files??{})])].sort();
      const differences=paths.filter(path=>row.files?.[path]!==row.expectedFiles?.[path]).map(path=>({path,actual:row.files?.[path]??'<missing>',expected:row.expectedFiles?.[path]??'<absent>'}));
      lines.push('File effects: '+(row.files?JSON.stringify(differences).slice(0,1000):'result unavailable; expected '+JSON.stringify(row.expectedFiles).slice(0,700)));
    }
    for(const step of row.modelTrace??[]){
      const calls=step.calls as {name:string;arguments?:Record<string,unknown>}[];
      lines.push('Action: '+calls.map(call=>call.name+' '+JSON.stringify(call.arguments??{}).slice(0,220)).join('; '));
      if(/error|rejected|failed/i.test(step.observation))lines.push('Tool observation: '+step.observation.slice(0,400));
    }
  }
  const result=lines.join('\n');
  return result.slice(0,6000)+(result.length>6000?'\n[card clipped; inspect the implicated sourceFiles or evidence field]':'');
}

/** Identify an evidenced objective; the model still decides which change can help. */
export function opportunity(policy:ImprovementPolicy,evidence:TrainingEvidence[]):{kind:'fixture'|'quality'|'efficiency'|'source-size'|'none';reason:string} {
  if(evidence.some(row=>row.failureKind==='fixture'))return {kind:'fixture',reason:'Independent fixture failed before target execution.'};
  if(evidence.some(row=>!row.passed))return {kind:'quality',reason:'A current target execution failed; diagnose its actual outcome and trace.'};
  if(policy.objective==='model-calls'&&evidence.some(row=>row.modelCalls!==undefined&&row.modelCalls>1))return {kind:'efficiency',reason:'Correct answers still took multiple model requests. Inspect the trace for avoidable inspection, repair, delegation or a separate completion request. Passing quality does not complete the cost objective.'};
  if(policy.objective==='source-size')return {kind:'source-size',reason:'Correct answers permit a behavior-preserving source simplification; test a real size reduction.'};
  return {kind:'none',reason:'No observed failure or measured cost opportunity supports this objective.'};
}
