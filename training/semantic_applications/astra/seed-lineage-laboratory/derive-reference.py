"""Private author reconstruction of the closed Lantern fixture; never model-visible or an NL execution claim."""
from pathlib import Path
from fractions import Fraction as F
from functools import lru_cache
import json,re
base=Path(__file__).resolve().parent/'lantern-bank';fr=base/'fixture';manifest=json.loads((fr/'manifest.json').read_text())
clauses={};coverage=[]
for d in manifest['documents']:
 ids=[]
 for line in (fr/d['path']).read_text().splitlines():
  m=re.match(r'\[([^]]+)\]',line)
  if m: clauses[m[1]]={'sourceId':d['id'],'path':d['path'],'quote':line};ids.append(m[1])
 coverage.append({'sourceId':d['id'],'clauseIds':ids})
w=lambda *ids:[clauses[x] for x in ids]
f=lambda x:{'numerator':F(x).numerator,'denominator':F(x).denominator}
masses={'D_AA':4,'D_Aa':12,'D_aa':2,'R_AA':3,'R_Aa':4,'R_aa':2,'M_AA':2,'M_Aa':1,'M_aa':1}
def world(t):
 theory,g=t.split('_');return {'worldId':t,'ticketId':t,'theoryId':theory,'assignments':[{'accessionId':'U','genotype':g,'cytoplasm':'amber'},{'accessionId':'N','genotype':'aa','cytoplasm':'amber'},{'accessionId':'T','genotype':'aa','cytoplasm':'clear'}],'priorMass':masses[t],'witnesses':w('J01',{'D':'J02','R':'J03','M':'J04'}[theory],'P03','P04')}
allw={t:world(t) for t in masses};survive=['D_Aa','R_aa','M_Aa','M_aa']
def weighted(tickets):
 total=sum(masses[t] for t in tickets);return [{'world':allw[t],'mass':f(masses[t]),'probability':f(F(masses[t],total))} for t in tickets]
rejected=[{'worldId':t,'ticketId':t,'reason':'Established aa mother cannot transmit A to an AA child.' if t.endswith('_AA') else 'Valid bright adult observation contradicts this theory and genotype.','witnesses':w('P02','I02') if t.endswith('_AA') else w('O01','H02' if t.startswith('D') else 'H03','A01')} for t in masses if t not in survive]
assess=[]
for obs in ['ADULT_U','COAT_U','WARM_U']:
 ts=[t for t in masses if not t.endswith('_AA')] if obs=='ADULT_U' else survive
 for t in ts:
  theory,g=t.split('_');bright=(theory=='D' and g!='aa') or (theory=='R' and g=='aa') or theory=='M'
  answer=('bright' if bright else 'dark') if obs=='ADULT_U' else 'coat:amber'
  support=w('O01','A01','H0'+str({'D':2,'R':3,'M':4}[theory])) if obs=='ADULT_U' else w('O02','I05','A04') if obs=='COAT_U' else w('O03','A05')
  valid=obs!='WARM_U';likelihood=F(int(bright)) if obs=='ADULT_U' else F(1)
  pred={'worldId':t,'itemId':obs,'answers':[{'answer':answer,'probability':f(1),'witnesses':support}] if valid else [],'usable':valid,'reason':'Actual biological quantity under its source protocol.' if valid else 'Warm sensitivity loss gives no phenotype evidence.','witnesses':support}
  assess.append({'observationId':obs,'worldId':t,'compatible':likelihood>0,'likelihood':f(likelihood),'prediction':pred,'reason':'Condition on valid adult expression.' if obs=='ADULT_U' else 'This record does not distinguish adult mechanism worlds.','witnesses':support})
posterior={'worlds':weighted(survive),'rejected':rejected,'assessments':assess,'unresolved':[]}
stockids=['U_FLOWER','T_POLLEN','T_FLOWER','U_POLLEN','U_LEAF','COAT_IMAGE_SLOT'];stock=[{'materialId':s,'quantity':1} for s in stockids]
expdata={'S':(1,'cross','U','T','COOL_UNIFORM',['U_FLOWER','T_POLLEN'],'E01'),'C':(3,'cross','T','U','COOL_COLOR',['T_FLOWER','U_POLLEN'],'E02'),'G':(2,'probe',None,None,'GENOTYPE',['U_LEAF'],'E03'),'F':(1,'image',None,None,'COAT_IMAGE',['COAT_IMAGE_SLOT'],'E04'),'W':(1,'cross','U','T','WARM_COLOR',['U_FLOWER','T_POLLEN'],'E05')}
experiments={e:{'experimentId':e,'slotId':'SLOT_'+e,'kind':d[1],'motherId':d[2],'fatherId':d[3],'subjectId':'U','assayId':d[4],'cost':f(d[0]),'materials':[{'materialId':s,'quantity':1} for s in d[5]],'witnesses':w(d[6],'K01','K02','K03')} for e,d in expdata.items()}
# Independent closed-source reconstruction: offspring frequencies derive from I02/I03, then H02-H04.
def answer(e,t):
 theory,g=t.split('_')
 if e=='G':return 'genotype:'+g
 if e=='F':return 'coat:amber'
 if e=='W':return None
 gametes=['A','a'] if g=='Aa' else ['a']
 genotypes=['Aa' if a=='A' else 'aa' for a in gametes];cyto='amber' if e=='S' else 'clear'
 bright=sum((theory=='D' and z!='aa') or (theory=='R' and z=='aa') or (theory=='M' and cyto=='amber') for z in genotypes)
 ratio=F(bright,len(genotypes))
 return ('mixed' if ratio not in [0,1] else 'uniform') if e=='S' else 'bright:'+str(ratio.numerator)+'/'+str(ratio.denominator)
partitions=[]
for e,exp in experiments.items():
 support=w(expdata[e][6],{'S':'A02','C':'A01','G':'A03','F':'A04','W':'A05'}[e],'I02','I03','I04','I05')
 cross={'experimentId':e,'motherId':exp['motherId'],'fatherId':exp['fatherId'],'subjectId':'U','compartment':'maternal seed coat' if e=='F' else 'nuclear genotype' if e=='G' else 'adult lamp','interpretable':True,'reason':'Actual original experiment direction and assayed tissue.','witnesses':support}
 preds=[{'worldId':t,'itemId':e,'answers':[{'answer':answer(e,t),'probability':f(1),'witnesses':support+w('H0'+str({'D':2,'R':3,'M':4}[t[0]]))}] if e!='W' else [],'usable':e!='W','reason':'Source-defined complete-tray or original-subject readout.' if e!='W' else 'Warm protocol explicitly invalid.','witnesses':support} for t in survive]
 branches=[]
 for a in sorted({answer(e,t) for t in survive} if e!='W' else []):
  members=[t for t in survive if answer(e,t)==a];branches.append({'answer':a,'worldLikelihoods':[{'worldId':t,'likelihood':f(int(t in members))} for t in survive],'probability':f(F(sum(masses[t] for t in members),16)),'goalClasses':sorted({t[0] for t in members}),'witnesses':support})
 partitions.append({'experiment':exp,'cross':cross,'predictions':preds,'branches':branches,'usable':e!='W','reason':'Valid source protocol.' if e!='W' else 'Invalid sensitivity regime.','witnesses':support})
nodes=[];transitions=[];values={};keyids={}
def ensure(ts,remaining,st):
 key=(tuple(sorted(ts)),tuple(sorted(remaining)),tuple(st))
 if key in keyids:return keyids[key]
 nid='belief-'+str(len(nodes));keyids[key]=nid;goals=sorted({t[0] for t in ts})
 node={'nodeId':nid,'worlds':weighted(ts),'remainingSlotIds':['SLOT_'+e for e in remaining],'stock':[{'materialId':s,'quantity':q} for s,q in zip(stockids,st)],'goalClasses':goals,'expanded':True,'terminal':'solved' if len(goals)==1 else 'open','unresolved':[]};nodes.append(node)
 if len(goals)==1:
  values[nid]={'nodeId':nid,'selectedExperimentId':None,'expectedCost':f(0),'worstCost':f(0),'candidates':[],'comparison':{'selectedExperimentId':None,'tiedExperimentIds':[],'reason':'Requested mechanism identified.','witnesses':w('Q01')},'solvedGoalClass':goals[0],'unresolved':[]};return nid
 candidates=[]
 for e in remaining:
  exp=experiments[e]
  if e=='W' or any(st[stockids.index(m['materialId'])]<m['quantity'] for m in exp['materials']):continue
  newst=list(st)
  for m in exp['materials']:newst[stockids.index(m['materialId'])]-=m['quantity']
  groups={a:[t for t in ts if answer(e,t)==a] for a in sorted({answer(e,t) for t in ts})};bs=[];cost=F(expdata[e][0]);worst=F(0)
  for a,members in groups.items():
   child=ensure(members,[x for x in remaining if x!=e],newst);p=F(sum(masses[t] for t in members),sum(masses[t] for t in ts));cv=values[child]
   cost+=p*F(cv['expectedCost']['numerator'],cv['expectedCost']['denominator']);worst=max(worst,F(cv['worstCost']['numerator'],cv['worstCost']['denominator']));bs.append({'answer':a,'probability':f(p),'toNodeId':child})
  transitions.append({'fromNodeId':nid,'experimentId':e,'use':{'experimentId':e,'feasible':True,'goalRelevant':len(groups)>1,'reason':'Original slot/material is available; actual answer partition supplies stated relevance.','witnesses':exp['witnesses']+w('Q01')},'branches':bs})
  candidates.append({'experimentId':e,'expectedCost':f(cost),'worstCost':f(F(expdata[e][0])+worst),'complete':True,'branches':[{'answer':b['answer'],'probability':b['probability'],'childNodeId':b['toNodeId']} for b in bs],'reason':'Complete branches consumed from actual child values.'})
 if not candidates:raise Exception('Unsolvable reference node '+nid)
 c=sorted(candidates,key=lambda c:(F(c['expectedCost']['numerator'],c['expectedCost']['denominator']),F(c['worstCost']['numerator'],c['worstCost']['denominator']),c['experimentId']))[0]
 values[nid]={'nodeId':nid,'selectedExperimentId':c['experimentId'],'expectedCost':c['expectedCost'],'worstCost':c['worstCost'],'candidates':candidates,'comparison':{'selectedExperimentId':c['experimentId'],'tiedExperimentIds':[x['experimentId'] for x in candidates if x['expectedCost']==c['expectedCost'] and x['worstCost']==c['worstCost']],'reason':'Expected cost, then worst cost, then original experiment ID.','witnesses':w('Q02')},'solvedGoalClass':None,'unresolved':[]}
 return nid
rootid=ensure(survive,list(experiments),[1]*6)
stockrules=[{'id':x,'text':clauses[x]['quote'],'witnesses':w(x)} for x in ['K01','K02','K03','K04']]
closure={'nodes':nodes,'transitions':transitions,'pendingNodeIds':[],'partitions':partitions,'stockRules':stockrules,'goal':'Identify adult lamp mechanism D, R or M.','finished':True,'unresolved':[]}
policy={'rootNodeId':rootid,'closure':closure,'values':list(values.values()),'expectedCost':values[rootid]['expectedCost'],'worstCost':values[rootid]['worstCost'],'unresolved':[]}
reachable=[];todo=[rootid]
while todo:
 n=todo.pop()
 if n in reachable:continue
 reachable.append(n);v=values[n]
 if v['selectedExperimentId']:
  tr=next(t for t in transitions if t['fromNodeId']==n and t['experimentId']==v['selectedExperimentId']);todo += [b['toNodeId'] for b in tr['branches']]
residual=[]
for n in nodes:
 if n['nodeId'] not in reachable or n['terminal']!='solved':continue
 residual.append({'nodeId':n['nodeId'],'mechanismKnown':True,'theoryIds':n['goalClasses'],'genotypeAlternatives':[{'accessionId':a,'genotypes':sorted({next(x['genotype'] for x in z['world']['assignments'] if x['accessionId']==a) for z in n['worlds']})} for a in ['U','N','T']],'reason':'Goal is mechanism; retain all consistent genotype alternatives.','witnesses':w('Q01','H04','P02')})
report={'caseId':manifest['caseId'],'posterior':posterior,'predictions':{'partitions':partitions,'unresolved':[]},'policy':policy,'audits':[{'subjectId':s,'passed':True,'reason':reason,'witnesses':w(*ids)} for s,reason,ids in [('prior','Joint masses preserved through observed evidence.',['J01','O01']),('compartments','Coat is independent maternal tissue; reciprocal cytoplasm follows actual mother.',['I03','I05','H05']),('warm','Invalid sensitivity supplies no phenotype distinction.',['A05','O03']),('stock','Hypothetical siblings retain separate material snapshots.',['K02','K03']),('objective','All positive-mass chosen branches identify mechanism at the declared cost optimum.',['Q01','Q02'])]],'residual':residual,'coverage':coverage,'unresolved':[]}
(base/'reference-certificate.json').write_text(json.dumps(report,indent=2)+'\n')
oracle={'schema':'natlang.seed-lineage-oracle/1','caseId':manifest['caseId'],'source_scope':'Closed fictional Lantern bank; independent finite reconstruction outside model-visible roots.','surviving_ticket_masses':{t:masses[t] for t in survive},'rejected_ticket_ids':[t for t in masses if t not in survive],'experiment_answers':{e:{t:answer(e,t) for t in survive} for e in experiments},'expected_root_cost':f(F(7,4)),'expected_worst_cost':f(4),'selected_root_experiment':'S','root_candidate_costs':{c['experimentId']:{'expected':c['expectedCost'],'worst':c['worstCost']} for c in values[rootid]['candidates']},'semantic_review_required':True,'training_admitted':False}
(base/'oracle.json').write_text(json.dumps(oracle,indent=2)+'\n')
print(json.dumps({'nodes':len(nodes),'transitions':len(transitions),'root_candidates':oracle['root_candidate_costs'],'selected':values[rootid]['selectedExperimentId'],'expected':policy['expectedCost'],'worst':policy['worstCost'],'chosen_reachable':len(reachable)}))
