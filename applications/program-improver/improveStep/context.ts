import {transformationText} from './transformations';
import type {ImprovementPolicy,EditRequest,Diagnosis,Hypothesis,TrainingEvidence,SourceFile,ExperimentFeedback} from '../types';

/** How much of each part the compact card shows; the complete text stays in the typed arguments. */
export const CLIPS={diagnostic:700,candidate:900,source:1200,error:400,observed:700,effects:1000,action:220,observation:400,card:6000};

/** What the edit stage reads: the hypothesis, its diagnosis and the source, with the compact card for orientation. */
export function request(policy:ImprovementPolicy,hypothesis:Hypothesis,diagnosis:Diagnosis,evidence:TrainingEvidence[],sourceFiles:SourceFile[],lastExperiment?:ExperimentFeedback):EditRequest {
  const serviceDeclarations=evidence.find(row=>row.serviceDeclarations)?.serviceDeclarations;
  return {brief:brief(policy,evidence,sourceFiles,lastExperiment),goal:policy.goal,mode:policy.mode,objective:policy.objective??'quality',hypothesis,diagnosis,sourceFiles,allowedFiles:policy.allowedFiles,
    ...(serviceDeclarations?{serviceDeclarations}:{}),...(policy.transformation?{transformation:transformationText(policy.transformation)}:{})};
}

/** A small first read; complete source and observations remain available in the typed context. */
export function brief(policy:ImprovementPolicy,evidence:TrainingEvidence[],sourceFiles:SourceFile[],lastExperiment?:ExperimentFeedback):string {
  const lines=['Goal: '+policy.goal,'Mode: '+policy.mode+'; objective: '+(policy.objective??'quality'),'Allowed source edits: '+policy.allowedFiles.join(', ')];
  if(lastExperiment){
    lines.push('Previous experiment: '+lastExperiment.reason);
    for(const diagnostic of lastExperiment.diagnostics??[])lines.push('Previous diagnostic: '+diagnostic.slice(0,CLIPS.diagnostic));
    for(const file of lastExperiment.sourceFiles.filter(file=>!sourceFiles.some(current=>current.path===file.path&&current.text===file.text)))lines.push('Previous candidate '+file.path+':\n'+file.text.slice(0,CLIPS.candidate));
  }
  for(const file of sourceFiles)lines.push('Source '+file.path+':\n'+file.text.slice(0,CLIPS.source)+(file.text.length>CLIPS.source?'\n[clipped; full text in sourceFiles]':''));
  for(const row of evidence){
    lines.push('Training '+(row.caseId??'case')+': '+(row.passed?'passed':'failed')+'; model requests '+(row.modelCalls??'unknown')+(row.failureKind?'; '+row.failureKind:'')+(row.error?'; '+row.error.slice(0,CLIPS.error):''));
    if(!row.passed)lines.push('Observed: '+JSON.stringify({args:row.args,value:row.value,expected:row.expected}).slice(0,CLIPS.observed));
    if(!row.passed&&row.expectedFiles){
      const paths=[...new Set([...Object.keys(row.expectedFiles),...Object.keys(row.files??{})])].sort();
      const differences=paths.filter(path=>row.files?.[path]!==row.expectedFiles?.[path]).map(path=>({path,actual:row.files?.[path]??'<missing>',expected:row.expectedFiles?.[path]??'<absent>'}));
      lines.push('File effects: '+(row.files?JSON.stringify(differences).slice(0,CLIPS.effects):'result unavailable; expected '+JSON.stringify(row.expectedFiles).slice(0,CLIPS.observed)));
    }
    for(const step of row.modelTrace??[]){
      const calls=step.calls as {name:string;arguments?:Record<string,unknown>}[];
      lines.push('Action: '+calls.map(call=>call.name+' '+JSON.stringify(call.arguments??{}).slice(0,CLIPS.action)).join('; '));
      if(/error|rejected|failed/i.test(step.observation))lines.push('Tool observation: '+step.observation.slice(0,CLIPS.observation));
    }
  }
  const result=lines.join('\n');
  return result.slice(0,CLIPS.card)+(result.length>CLIPS.card?'\n[card clipped; inspect the implicated sourceFiles or evidence field]':'');
}
