"""Private authored source witness; never included in model-visible project or fixture."""
from pathlib import Path
from copy import deepcopy
import json,re
base=Path(__file__).parent;case=base/'glass-harbor';fixture=case/'fixture'
manifest=json.loads((fixture/'manifest.json').read_text());clauses={};coverage=[]
for d in manifest['documents']:
 ids=[]
 for line in (fixture/d['path']).read_text().splitlines():
  m=re.match(r'^\[([^]]+)\]',line)
  if m:clauses[m[1]]={'sourceId':d['id'],'path':d['path'],'quote':line};ids.append(m[1])
 coverage.append({'sourceId':d['id'],'clauseIds':ids})
def ws(*ids):return [deepcopy(clauses[x]) for x in ids]
spec={'B1':('Bell','Dock',0,4,'I','raise','A01'),'B2':('Bell','Dock',1,4,'I','none','A02'),'B3':('Bell','Dock',2,7,'II','natural','A03'),'R1':('Reed','Dock',0,2,'O','none','A04'),'B4':('Bell','Channel',0,4,'II','none','A05'),'B5':('Bell','Channel',1,5,'II','none','A06'),'R2':('Reed','Channel',0,2,'II','none','A07'),'R3':('Reed','Channel',1,3,'II','none','A08')}
def target(claim,event,basis,applicable,evidence):return {'basis':basis,'claimId':claim,'eventId':event,'applicable':applicable,'reason':'Source authority and attachment retained independently.','witnesses':ws(*evidence)}
def option(e,chosen=None):
 voice,sec,order,rank,glyph,acc,clause=spec[e];applied=[];assign=[];decisions=[];support=[clause,'N01','N02','N03','E01']
 if e=='B2':applied=['S01'];assign=[{'claimId':'S01','targetEventId':'B2'}];decisions=[target('S01',e,'established-target',True,['S01','E01'])];support+=['S01']
 if e in ['B5','R3']:
  decisions=[target('S02',e,'finite-exclusive-candidate',None,['S02','E03'])];assign=[{'claimId':'S02','targetEventId':chosen}];support+=['S02','E03']
  if e==chosen:rank-=1;applied=['S02']
 event={'readingId':'reading-'+e,'eventId':e,'sourceId':'AUTO','voiceId':voice,'sectionId':sec,'order':order,'rank':rank,'durationGlyph':glyph,'accidental':acc,'kind':'attack','witnesses':ws(*support)}
 return {'targetDecisions':decisions,'optionId':'option-'+e+('-'+chosen if chosen else ''),'event':event,'claimAssignments':assign,'appliedClaimIds':applied,'reason':'Chosen underlying autograph plus exactly the source-assigned authorized operation.','witnesses':ws(*support)}
bundles=[]
for e in spec:
 opts=[option(e,t) for t in ['B5','R3']] if e in ['B5','R3'] else [option(e)]
 bundles.append({'eventId':e,'options':opts,'rejectedReadingIds':['uncorrected-A02','copy-C11','copy-C21'] if e=='B2' else [],'reason':'Authorized duration correction defeats inherited long reading; ambiguous target alternatives remain.','witnesses':ws(spec[e][-1],*(['S01','H01'] if e=='B2' else ['E01']))})
families=[{'familyId':'notation-chain','sourceIds':['AUTO','COPY1','COPY2'],'scope':'Notation dependence through documented exemplar chain.','witnesses':ws('H01')},{'familyId':'observation-chain','sourceIds':['REHEARSAL','REHEARSAL_COPY'],'scope':'One independent rehearsal and its copied report.','witnesses':ws('H03','TC01')}]
used={s for f in families for s in f['sourceIds']}
for d in manifest['documents']:
 if d['id'] not in used:
  ids=next(x['clauseIds'] for x in coverage if x['sourceId']==d['id']);families.append({'familyId':'family-'+d['id'],'sourceIds':[d['id']],'scope':'Separate original documentary act; authority is assessed by source policy.','witnesses':ws(ids[0])})
editions=[]
for chosen in ['B5','R3']:
 choices=[option(e,chosen if e in ['B5','R3'] else None) for e in spec];hyp={'hypothesisId':'edition-'+chosen,'choices':choices,'claimAssignments':[{'claimId':'S01','targetEventId':'B2'},{'claimId':'S02','targetEventId':chosen}],'witnesses':ws('S01','S02','E03')};events=[]
 for occ,sec,start in [('D1','Dock',0),('C1','Channel',4),('D2','Dock',8)]:
  for voice in ['Bell','Reed']:
   state={};elapsed=0
   for op in sorted((x for x in choices if x['event']['sectionId']==sec and x['event']['voiceId']==voice),key=lambda x:x['event']['order']):
    e=op['event'];before=[{'rank':k,'offset':v} for k,v in sorted(state.items())]
    if e['accidental']=='raise':state[e['rank']]=1
    elif e['accidental']=='natural':state[e['rank']]=0
    duration={'I':1,'II':2,'O':4}[e['durationGlyph']];rank=e['rank']+state.get(e['rank'],0)+(2 if voice=='Reed' else 0)
    events.append({'eventId':e['eventId'],'occurrenceId':occ,'sectionId':sec,'voiceId':voice,'onset':start+elapsed,'durationTicks':duration,'soundingRank':rank,'attacks':True,'beforeAccidentals':before,'afterAccidentals':[{'rank':k,'offset':v} for k,v in sorted(state.items())],'readingOptionId':op['optionId'],'reason':'Actual chosen reading, local carried setting and one instrument offset.','witnesses':op['witnesses']+ws('N04')});elapsed+=duration
   assert elapsed==4
 findings=[]
 for occ in ['D1','D2']:findings.append({'hypothesisId':hyp['hypothesisId'],'constraintId':'T01','occurrenceId':occ,'passed':True,'reason':'Both complete voices end at four ticks.','witnesses':ws('T01','S01')})
 for cid in ['T02','T03']:findings.append({'hypothesisId':hyp['hypothesisId'],'constraintId':cid,'occurrenceId':'C1','passed':True,'reason':'Actual attacks or unordered final sonority satisfy the observed relation.','witnesses':ws(cid,'S02')})
 for occ in ['D1','C1','D2']:findings.append({'hypothesisId':hyp['hypothesisId'],'constraintId':'N01','occurrenceId':occ,'passed':True,'reason':'Each required voice has complete continuous source events over four ticks.','witnesses':ws('N01','P02')})
 editions.append({'edition':{'hypothesis':hyp,'events':events,'unresolved':[]},'findings':findings,'admissible':True})
r={'caseId':'glass-harbor','bundleOrder':list(spec),'families':families,'bundles':bundles,'editions':editions,'rejected':[{'hypothesisId':'partial-target-'+prior,'eventId':'R3','priorOptionIds':[option(e,prior if e=='B5' else None)['optionId'] for e in spec if e!='R3'],'priorAssignments':[{'claimId':'S01','targetEventId':'B2'},{'claimId':'S02','targetEventId':prior}],'incomingOptionId':option('R3',incoming)['optionId'],'incomingAssignments':[{'claimId':'S02','targetEventId':incoming}],'reason':'The actual prior S02 target conflicts with the actual incoming option assignment.','witnesses':ws('S02','E03')} for prior,incoming in [('B5','R3'),('R3','B5')]],'ambiguities':[{'claimId':'S02','targetEventIds':['B5','R3'],'differingEventIds':['B5','R3'],'reason':'Authenticated exclusive attachment remains unresolved; unordered rehearsal sonority supports both voice assignments.','witnesses':ws('S02','T03','E03','E04','X01')}],'coverage':coverage,'unresolved':[]}
(case/'reference-certificate.json').write_text(json.dumps(r,indent=2)+'\n')
print('two independently specified source target assignments; each has12 actual performed events')
