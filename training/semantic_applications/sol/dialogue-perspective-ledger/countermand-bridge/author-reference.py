"""Private authored certificate assembly from explicit semantic annotations.
This file is outside visible roots. It does not interpret source text or execute NL callables.
Typed state copying, quote offsets, sorting and original-ID binding are exact mechanisms.
"""
from pathlib import Path
import json,re,copy
ROOT=Path(__file__).parent
snapshot=json.loads((ROOT/'snapshot.json').read_text());request=json.loads((ROOT/'request.json').read_text())
clone=copy.deepcopy
moment=lambda d,h:{'day':d,'hour':h}
tick=lambda t:t['day']*24+t['hour']
clauses={}
for doc in snapshot['documents']:
 for line in doc['text'].splitlines():
  m=re.match(r'^\[([^]]+)\]',line)
  if m:clauses[m[1]]={'clauseId':m[1],'sourceId':doc['sourceId'],'path':doc['path'],'text':line}
def witnesses(*ids):return [{**{k:clauses[i][k] for k in ['clauseId','sourceId','path']},'quote':clauses[i]['text']} for i in ids]
# Original authenticated metadata and actual speech content, hand-authored from the visible packet.
turn_annotations=[
('P17','Oren',17,10,['Sada','Tavi'],'P','shared'),
('W18','Oren',18,0,['Sada','Tavi'],None,'shared'),
('S18','Sada',18,2,[],None,'private'),
('T18_DOUBT','Tavi',18,2.5,[],None,'private'),
('T18_ADOPT','Tavi',18,4,[],None,'private'),
('N18','Oren',18,5,['Sada'],'P','shared'),
('B18','Sada',18,6,['Oren'],'P','shared'),
('B19','Sada',19,9,['Oren'],'P','shared'),
('C18','Noor',18,8,['Sada'],None,'shared'),
('S19_EARLY','Sada',19,8,[],None,'private'),
('C19','Noor',19,10.5,['Sada'],None,'shared'),
('O19_REPORT','Oren',19,12,['Remy'],None,'shared'),
('REM19','Remy',19,13,['Tavi'],None,'shared'),
('O19_ADOPT','Oren',19,12.5,[],None,'private')]
turns=[]
for ident,actor,d,h,audience,channel,visibility in turn_annotations:
 # Quotation delimiters mechanically extract the explicitly selected whole original turn.
 text=clauses[ident]['text'];start=text.index('"');end=text.rindex('"');content=text[start+1:end]
 turns.append({'turnId':ident,'clauseId':ident,'speaker':actor,'occurredAt':moment(d,h),'audienceIds':audience,'channelId':channel,'visibility':visibility,'content':content,'witnesses':witnesses(ident)})
byturn={t['turnId']:t for t in turns}
props={'P_COLLECT18':'Noor collected on day18','P_COLLECT20':'Noor collected before day20 noon','P_ACK_B2':'Sada personally acknowledged B2','P_ACK_B3':'Sada personally acknowledged B3','P_KEY19':'Oren released key to Sada on day19','P_KEY20':'Oren released key to Sada on day20','P_CODE20':'Oren supplied code to Tavi on day20','P_ALL_ADOPT':'Both named original beneficiaries adopted W18','P_WITHDRAW_RECEIVED_TAVI':'Tavi received W18'}
physical_annotations=[('F_COLLECTION','P_COLLECT18',False,None,'F01'),('F_COLLECTION','P_COLLECT20',True,moment(19,10),'F01'),('F_KEY','P_KEY19',True,moment(19,11),'F02'),('F_KEY','P_KEY20',False,None,'F02'),('F_CODE','P_CODE20',False,None,'F03'),('F_BRIEF','P_ACK_B2',True,moment(18,6),'F04'),('F_BRIEF','P_ACK_B3',True,moment(19,9),'F04'),('F_ADOPTIONS','P_ALL_ADOPT',True,moment(18,4),'F05')]
physical=[{'factId':i,'propositionId':p,'value':v,'occurredAt':at,'completenessThrough':moment(21,12),'witnesses':witnesses(c)} for i,p,v,at,c in physical_annotations]
receipt_annotations=[('HP17',['P17'],'Sada',moment(17,10),moment(17,10),False,[]),('HP17',['P17'],'Tavi',moment(17,10),moment(17,10),False,[]),('HW18_S',['W18'],'Sada',moment(18,0),moment(18,1),False,[]),('RW18_T',['W18'],'Tavi',moment(18,0),moment(18,3),False,[]),('HN18',['N18'],'Sada',moment(18,5),moment(18,5),False,[]),('HB19',['B18'],'Oren',moment(18,6),moment(18,6),False,[]),('HB19',['B19'],'Oren',moment(19,9),moment(19,9),False,[]),('HC18',['C18'],'Sada',moment(18,8),moment(18,8),False,[]),('RC19_S',['C19'],'Sada',moment(19,10.5),moment(20,1),True,['F_COLLECTION']),('RADOPT_O',['S18','T18_ADOPT'],'Oren',moment(19,12),moment(19,12),True,['F_ADOPTIONS']),('HREM19',['REM19','W18'],'Tavi',moment(19,13),moment(19,13),False,[])]
receipts=[{'receiptId':i,'turnIds':ids,'recipientId':who,'dispatchedAt':sent,'receivedAt':arrived,'readAt':arrived,'verifiedEvidence':verified,'verifiedFactIds':facts,'witnesses':witnesses(i)} for i,ids,who,sent,arrived,verified,facts in receipt_annotations]
actors=['Oren','Sada','Tavi','Noor','Remy']
source={'caseId':snapshot['caseId'],'clauses':list(clauses.values()),'rules':[c for k,c in clauses.items() if re.fullmatch(r'R\d+',k)],'turns':turns,'propositions':[{'propositionId':k,'meaning':v,'witnesses':witnesses('P01')} for k,v in props.items()],'physicalFacts':physical,'temporal':{'entries':[{'turnId':t['turnId'],'originalTime':t['occurredAt'],'reportingTime':t['occurredAt'] if t['turnId'] in ['O19_REPORT','REM19'] else None,'reason':'Authenticated original turn; embedded W18 remains day18 hour0.','witnesses':t['witnesses']} for t in turns],'unresolved':[]},'participants':{'participantIds':actors,'memberships':[{'participantId':p,'channelId':'P','from':moment(17,0),'through':moment(21,24),'witnesses':witnesses('C02')} for p in ['Oren','Sada','Tavi']],'unresolved':[]},'receipts':{'receipts':receipts,'unresolved':[]},'unresolved':[]}
def predicate(pid,role,prop,cutoff=None,strict=False,subject=None,obj=None,original='P17'):
 return {'predicateId':pid,'role':role,'propositionId':prop,'cutoff':cutoff,'strictBefore':strict,'subjectId':subject,'objectId':obj,'reason':'Authored original actor/version/day scope, independent of reported occurrence.','witnesses':witnesses(original,'R08','R09')}
oldcollect=predicate('P17_COLLECT','activation','P_COLLECT20',moment(20,12),True,'Noor','crate');newcollect=predicate('N18_COLLECT','activation','P_COLLECT20',moment(20,12),True,'Noor','crate','N18');newack=predicate('N18_ACK','activation','P_ACK_B3',None,False,'Sada','B3','N18');oldkey=predicate('P17_KEY','fulfillment','P_KEY20',moment(20,None),False,'Oren','key');oldcode=predicate('P17_CODE','fulfillment','P_CODE20',moment(20,None),False,'Oren','code');newkey=predicate('N18_KEY','fulfillment','P_KEY19',moment(19,None),False,'Oren','key','N18')
# Every role/claim/scope below is an authored conclusion, not an executable semantic classifier.
meanings={
'P17':('promise',True,[],['P_COLLECT20','P_KEY20','P_CODE20'],'Oren',['Sada','Tavi'],moment(20,None),[oldcollect],[oldkey,oldcode],[],False),
'W18':('retraction',True,[],[],None,[],None,[],[],['P17'],True),
'S18':('acknowledgment',True,[],[],None,[],None,[],[],[],True),
'T18_DOUBT':('doubt',True,[('P_ALL_ADOPT',None)],['P_ALL_ADOPT'],None,[],None,[],[],[],False),
'T18_ADOPT':('acknowledgment',True,[],[],None,[],None,[],[],[],True),
'N18':('promise',True,[],['P_COLLECT20','P_ACK_B3','P_KEY19'],'Oren',['Sada'],moment(19,None),[newcollect,newack],[newkey],[],False),
'B18':('acknowledgment',True,[('P_ACK_B2',True),('P_ACK_B3',False)],['P_ACK_B2','P_ACK_B3'],None,[],None,[],[],[],True),
'B19':('acknowledgment',True,[('P_ACK_B3',True)],['P_ACK_B3'],None,[],None,[],[],[],True),
'C18':('assertion',True,[('P_COLLECT18',True)],['P_COLLECT18'],None,[],None,[],[],[],True),
'S19_EARLY':('assertion',True,[('P_COLLECT18',True),('P_COLLECT20',True)],['P_COLLECT18','P_COLLECT20'],None,[],None,[],[],[],True),
'C19':('correction',True,[('P_COLLECT18',False),('P_COLLECT20',True)],['P_COLLECT18','P_COLLECT20'],None,[],None,[],[],['C18'],True),
'O19_REPORT':('report',True,[],['P_ALL_ADOPT'],None,[],None,[],[],[],False),
'REM19':('report',True,[],[],None,[],None,[],[],[],False),
'O19_ADOPT':('assertion',True,[('P_ALL_ADOPT',True)],['P_ALL_ADOPT'],None,[],None,[],[],[],True)}
trees=[];acts=[];references=[]
for t in turns:
 tid=t['turnId'];root={'span':{'spanId':'S_'+tid,'turnId':tid,'parentSpanId':None,'start':0,'end':len(t['content']),'inheritedSpeaker':t['speaker'],'inheritedTime':t['occurredAt']},'speaker':t['speaker'],'originalTime':t['occurredAt'],'originalTurnId':tid,'mode':'own','childSpanIds':[],'witnesses':t['witnesses'],'unresolved':[]};nodes=[root]
 if tid=='REM19':
  content='I withdraw my original promise P17.';assert content in byturn['W18']['content'];start=t['content'].index(content);sid='S_REM19_W18';root['childSpanIds'].append(sid);nodes.append({'span':{'spanId':sid,'turnId':tid,'parentSpanId':root['span']['spanId'],'start':start,'end':start+len(content),'inheritedSpeaker':'Oren','inheritedTime':moment(18,0)},'speaker':'Oren','originalTime':moment(18,0),'originalTurnId':'W18','mode':'quoted','childSpanIds':[],'witnesses':t['witnesses'],'unresolved':[]})
 trees.append({'turnId':tid,'nodes':nodes,'unresolved':[]});references.append({'turnId':tid,'bindings':[],'unresolved':[]})
 for i,n in enumerate(nodes):
  spec=meanings[n['originalTurnId']];kind,performed,claims,ps,issuer,ben,due,conditions,consequences,supersedes,endorsement=spec
  if i:performed=False;claims=[];endorsement=False
  meaning={'performed':performed,'kind':kind,'propositionIds':ps,'claims':[{'propositionId':p,'value':v} for p,v in claims],'issuerId':issuer,'beneficiaryIds':ben,'due':due,'conditions':clone(conditions),'consequences':clone(consequences),'supersedesTurnIds':supersedes,'endorsement':endorsement,'alternatives':[],'reason':'Actual authenticated force and selected original revision/promise scope; quoted depiction performs no fresh revision.','quotes':[t['content'][n['span']['start']:n['span']['end']]]}
  acts.append({'actId':'A_'+tid+'_'+str(i),'turnId':tid,'spanId':n['span']['spanId'],'originalTurnId':n['originalTurnId'],'speaker':n['speaker'],'occurredAt':n['originalTime'],'audienceIds':t['audienceIds'] if i==0 else [],'meaning':meaning,'witnesses':t['witnesses']})
primary={t['turnId']:next(a for a in acts if a['turnId']==t['turnId']) for t in turns};byact={a['actId']:a for a in acts}
scoped={'quoteTrees':trees,'references':references,'acts':acts,'scopeFindings':[{'findingId':'SC_'+t['turnId'],'kind':'original-lineage','passed':True,'originalIds':[t['turnId']],'reason':'Actual source force, time, adoption target and report scope are retained.','witnesses':t['witnesses']} for t in turns],'unresolved':[]}
# Remaining episode/evolution/proof annotations are added before any reference is emitted.
# Explicit episode facts and predicate values are hand-authored source conclusions.
episode_units={}
def event(at):
 return episode_units.setdefault(tick(at),{'episodeId':'E_'+str(at['day'])+'_'+str(at['hour']).replace('.','p'),'at':at,'kind':'utterance','turnIds':[],'actIds':[],'factIds':[],'availability':[],'witnesses':[]})
def availability(ep,actor,ids,receipt_ids,facts,cids):
 row={'participantId':actor,'turnIds':ids,'actIds':[a['actId'] for a in acts if a['turnId'] in ids],'receiptIds':receipt_ids,'verifiedFactIds':facts,'availableAt':ep['at'],'reason':'Actual original speaker access or documented receipt; evidence availability does not imply adoption.','witnesses':witnesses(*cids)};ep['availability'].append(row);ep['witnesses']+=row['witnesses']
for t in turns:
 ep=event(t['occurredAt']);ep['turnIds'].append(t['turnId']);ep['actIds'].append(primary[t['turnId']]['actId']);ep['witnesses']+=t['witnesses'];availability(ep,t['speaker'],[t['turnId']],[],[],[t['turnId']])
for r in receipts:
 ep=event(r['receivedAt']);ep['kind']='receipt' if not ep['turnIds'] else ep['kind'];availability(ep,r['recipientId'],r['turnIds'],[r['receiptId']],r['verifiedFactIds'],[r['receiptId']])
for at,ids,cids in [(moment(18,4),['F_ADOPTIONS'],['F05']), (moment(18,6),['F_BRIEF'],['F04']),(moment(19,9),['F_BRIEF'],['F04']),(moment(19,10),['F_COLLECTION'],['F01']),(moment(19,11),['F_KEY'],['F02']),(moment(21,0),['F_CODE'],['F03'])]:
 ep=event(at);ep['kind']='physical';ep['factIds']=ids;ep['witnesses']+=witnesses(*cids)
for at,cid in [(moment(20,0),'R09'),(moment(20,12),'R08'),(moment(21,0),'R09')]:
 ep=event(at);ep['kind']='boundary' if not ep['turnIds'] and not ep['factIds'] else ep['kind'];ep['witnesses']+=witnesses(cid)
episodes=[episode_units[k] for k in sorted(episode_units)];plan={'episodes':episodes,'unresolved':[]}
views=[{'participantId':p,'availableTurnIds':[],'availableActIds':[],'availableVerifiedFactIds':[],'beliefs':[],'commitments':[],'revisionIds':[],'adoptedRevisionActIds':[]} for p in actors];view_by_id={v['participantId']:v for v in views};history=[];deltas=[]
endorsement_journal={
'W18':('endorse',['W18'],[],[]),
'S18':('endorse',['W18'],['W18'],[]),
'T18_DOUBT':('doubt',['T18_DOUBT'],[],[]),
'T18_ADOPT':('endorse',['W18'],['W18'],[]),
'B18':('endorse',['B18'],[],[]),
'B19':('endorse',['B19'],[],[]),
'C18':('endorse',['C18'],[],[]),
'S19_EARLY':('endorse',['C18'],[],[]),
'C19':('endorse',['C19'],['C19'],['C18']),
'O19_ADOPT':('endorse',['S18','T18_ADOPT'],[],[])}
# Actual interpreted values at event boundaries; selecting the last annotation is exact timeline plumbing.
predicate_actual_journal={
'P17_COLLECT':[(moment(17,10),None),(moment(19,10),True)],
'N18_COLLECT':[(moment(18,5),None),(moment(19,10),True)],
'N18_ACK':[(moment(18,5),False),(moment(19,9),True)],
'P17_KEY':[(moment(17,10),None),(moment(21,0),False)],
'P17_CODE':[(moment(17,10),None),(moment(21,0),False)],
'N18_KEY':[(moment(18,5),None),(moment(19,11),True)]}
status_journal={'P17':[(moment(17,10),'conditional'),(moment(18,4),'retracted')],'N18':[(moment(18,5),'conditional'),(moment(19,10),'active'),(moment(19,11),'fulfilled')]}
def last_annotation(rows,at):return next(value for when,value in reversed(rows) if tick(when)<=tick(at))
def condition_reading(pred,ep,beneficiaries):
 actual=last_annotation(predicate_actual_journal[pred['predicateId']],ep['at']);values=[]
 for who in beneficiaries:
  rows=[b for b in view_by_id[who]['beliefs'] if b['status']=='endorsed' and b['propositionId']==pred['propositionId']];values.append(rows[-1]['value'] if rows else None)
 perceived=values[0] if values and all(v==values[0] for v in values) else None
 return {'predicateId':pred['predicateId'],'actual':actual,'issuerKnows':False,'beneficiaryBelieves':perceived,'reason':'Authored actual predicate timeline; issuer has no predicate endorsement with available verified support. Divergent beneficiary views stay null, with actual individual views retained in history.','witnesses':pred['witnesses']+witnesses('R05','R08','R09')}
for ep in episodes:
 ed={'episodeId':ep['episodeId'],'endorsements':[],'beliefs':[],'commitments':[],'revisions':[],'availability':clone(ep['availability'])}
 for row in ep['availability']:
  v=view_by_id[row['participantId']]
  for key,items in [('availableTurnIds',row['turnIds']),('availableActIds',row['actIds']),('availableVerifiedFactIds',row['verifiedFactIds'])]:v[key]=list(dict.fromkeys(v[key]+items))
 for tid in ep['turnIds']:
  a=primary[tid];who=a['speaker'];v=view_by_id[who];ann=endorsement_journal.get(tid)
  posture,basis,adopts,defeats=ann if ann else ('report-only' if a['meaning']['kind']=='report' else 'none',[],[],[])
  basis=[primary[i]['actId'] for i in basis];adopts=[primary[i]['actId'] for i in adopts];defeats=[primary[i]['actId'] for i in defeats]
  claims=clone(a['meaning']['claims']) if ann else []
  decision={'participantId':who,'actId':a['actId'],'claims':claims,'posture':posture,'basisActIds':basis,'defeatsActIds':defeats,'adoptsActIds':adopts,'reason':'Explicit source endorsement/adoption or report-only posture; other actors are not credited with the same endorsement.','quotes':[byturn[tid]['content']]};ed['endorsements'].append(decision)
  v['adoptedRevisionActIds']=list(dict.fromkeys(v['adoptedRevisionActIds']+adopts));bd={'participantId':who,'add':[],'defeatBeliefIds':[],'unresolved':[]}
  for old in v['beliefs']:
   if old['status']=='endorsed' and set(old['sourceActIds'])&set(defeats):bd['defeatBeliefIds'].append(old['beliefId']);old['status']='defeated'
  for i,c in enumerate(claims):
   corroboration=[f['factId'] for f in physical if f['propositionId']==c['propositionId'] and f['value']==c['value'] and f['factId'] in v['availableVerifiedFactIds']]
   b={'beliefId':'B_'+tid+'_'+str(i),'participantId':who,'propositionId':c['propositionId'],'value':c['value'],'status':'doubted' if posture=='doubt' else 'endorsed','acquiredAt':ep['at'],'sourceActIds':list(dict.fromkeys([a['actId']]+basis)),'availabilityEpisodeIds':[ep['episodeId']],'verifiedFactIds':list(dict.fromkeys(corroboration)),'witnesses':a['witnesses']};bd['add'].append(clone(b));v['beliefs'].append(b)
  if bd['add'] or bd['defeatBeliefIds']:ed['beliefs'].append(bd)
  if tid in ['P17','N18']:
   cm={'commitmentId':'K_'+tid,'issuerId':'Oren','beneficiaryIds':a['meaning']['beneficiaryIds'],'sourceActId':a['actId'],'propositionIds':[p['propositionId'] for p in a['meaning']['consequences']],'due':a['meaning']['due'],'conditions':[],'fulfillment':[],'status':'conditional','establishedAt':ep['at'],'witnesses':a['witnesses']};v['commitments'].append(cm);ed['commitments'].append({'participantId':'Oren','add':[clone(cm)],'replace':[],'unresolved':[]})
  # Explicit source-authored revision conclusions; this is not a semantic revision selector.
  if tid in ['W18','S18','T18_ADOPT','C19']:
   full=tid=='T18_ADOPT';rd={'revisionId':'REV_'+tid if full or tid=='C19' else None,'participantId':who,'commitmentOwnerId':'Oren' if full else None,'adoptingParticipantIds':['Sada','Tavi'] if full else ([] if tid=='W18' else [who]),'supersededActIds':[primary['C18']['actId']] if tid=='C19' else [primary['P17']['actId']],'defeatBeliefIds':bd['defeatBeliefIds'],'retractCommitmentIds':['K_P17'] if full else [],'reason':'Last actual beneficiary adoption retracts only issuer-owned P17; collection correction defeats only Noor original C18 endorsement; partial audience adoption applies no global retraction.','witnesses':witnesses(tid,'R06','R07'),'unresolved':[]};ed['revisions'].append(rd)
   if rd['revisionId']:v['revisionIds'].append(rd['revisionId'])
 for cm in view_by_id['Oren']['commitments']:
  original=byact[cm['sourceActId']]['turnId'];a=primary[original];cm['conditions']=[condition_reading(p,ep,cm['beneficiaryIds']) for p in a['meaning']['conditions']];cm['fulfillment']=[condition_reading(p,ep,cm['beneficiaryIds']) for p in a['meaning']['consequences']];cm['status']='conditional' if original=='P17' and ep['at']==moment(18,4) else last_annotation(status_journal[original],ep['at'])
 if view_by_id['Oren']['commitments']:ed['commitments'].append({'participantId':'Oren','add':[],'replace':clone(view_by_id['Oren']['commitments']),'unresolved':[]})
 for rd in ed['revisions']:
  if rd['commitmentOwnerId']:
   for cm in view_by_id[rd['commitmentOwnerId']]['commitments']:
    if cm['commitmentId'] in rd['retractCommitmentIds']:cm['status']='retracted'
 history.append({'episodeId':ep['episodeId'],'at':ep['at'],'views':clone(views)});deltas.append(ed)
evolution={'views':clone(views),'history':history,'deltas':deltas,'unresolved':[]}
answer_support={
'Q01':['P17','W18','HW18_S','S18','R06'],
'Q02':['P17','RW18_T','T18_DOUBT','R04','R06'],
'Q03':['P17','S18','T18_ADOPT','F05','R06'],
'Q04':['RADOPT_O','R05','R06'],
'Q05':['P17','F01','T18_ADOPT','R07'],
'Q06':['N18','B18','S19_EARLY','F01','F04','R08'],
'Q07':['N18','B19','F01','F04','R08'],
'Q08':['N18','F02','R09'],
'Q09':['P17','N18','F02','R07','R09'],
'Q10':['F02','P17','R09'],
'Q11':['S19_EARLY','HC18','R04'],
'Q12':['S19_EARLY','RC19_S','F01','R05'],
'Q13':['S19_EARLY','RC19_S','R04'],
'Q14':['O19_REPORT','RADOPT_O','R04','R05'],
'Q15':['O19_ADOPT','RADOPT_O','F05','R05'],
'Q16':['REM19','W18','HREM19','R03'],
'Q17':['B19','F04','R08'],
'Q18':['P17','F03','R09']}
answer_explanations={
'Q01':'Only Sada has explicitly adopted W18; all original beneficiaries must adopt before P17 retracts.',
'Q02':'Tavi actual arrival supplies access without adoption; the earlier explicit doubt does not adopt W18.',
'Q03':'Tavi explicit adoption joins Sada and applies terminal retraction to Oren original P17 ledger.',
'Q04':'Normative joint adoption occurred, but its verified records have not reached Oren and no Oren endorsement exists.',
'Q05':'Actual collection changes condition evidence, while source-terminal retraction prevents P17 revival.',
'Q06':'Fresh N18 has no actual collection yet; Sada belief does not fulfill an event condition, and B2 is not B3.',
'Q07':'The actual day19 collection and prior personal B3 acknowledgment activate fresh N18 independently.',
'Q08':'The actual day19 key release fulfills the fresh on-day consequent for its sole beneficiary Sada.',
'Q09':'Day19 delivery under N18 neither revives P17 nor fulfills P17 day20 and second-beneficiary scopes.',
'Q10':'The complete log establishes only a day19 release, with no key release on day20.',
'Q11':'Sada explicitly endorses C18 day18 claim in her signed note, although its physical proposition is false.',
'Q12':'Sada endorsement has no available supporting verified evidence; false endorsed belief is not qualifying knowledge.',
'Q13':'Later receipt exposes the correction and verified log, but Sada never adopts a defeating correction in this packet.',
'Q14':'Oren has now received verified adoption records, but expressly reports their text without endorsing joint adoption.',
'Q15':'Oren later explicit endorsement combines with available verified F_ADOPTIONS and qualifies knowledge.',
'Q16':'Remy performs a report of Oren original W18; the embedded withdrawal is depicted rather than newly performed.',
'Q17':'The actual personal B3 acknowledgment is a different version-specific event from B18 acknowledgment of B2.',
'Q18':'The complete day20 code log establishes no promised delivery to Tavi, without altering terminal retraction.'}
# Separately authored query projections, checked against the independent draft table rather than read as answers.
answer_projection_annotations={
'Q01':{'commitmentStatus':'conditional'},'Q02':{'commitmentStatus':'conditional'},
'Q03':{'commitmentStatus':'retracted'},'Q04':{'knowledgeValue':False},
'Q05':{'commitmentStatus':'retracted'},'Q06':{'commitmentStatus':'conditional'},
'Q07':{'commitmentStatus':'active','actorIds':['Oren']},'Q08':{'commitmentStatus':'fulfilled'},
'Q09':{'commitmentStatus':'retracted'},'Q10':{'truthValue':False},'Q11':{'beliefValue':True},
'Q12':{'knowledgeValue':False},'Q13':{'beliefValue':True},'Q14':{'knowledgeValue':False},
'Q15':{'knowledgeValue':True},'Q16':{'performed':False},'Q17':{'truthValue':True},'Q18':{'truthValue':False}}
assert answer_projection_annotations==json.loads((ROOT/'oracle.json').read_text())['query_observables']
answers=[]
for query in request['queries']:
 qid=query['queryId'];support=answer_support[qid];projection=clone(answer_projection_annotations[qid])
 who=query['perspective'];eligible=[h for h in history if tick(h['at'])<=tick(query['asOf'])];view=next(v for v in eligible[-1]['views'] if v['participantId']==who) if who else None
 target=query['targetId'];pids=[target] if target in props else [];aids=[primary[target]['actId']] if target in primary else []
 focus={'query':clone(query),'participantId':who,'asOf':clone(query['asOf']),'propositionIds':pids,'actIds':aids,'view':clone(view),'kind':query['dimension'],'reason':'Actual source target lineage, dimension and at-or-before participant view.','witnesses':witnesses(*support),'unresolved':[]}
 root='PF_'+qid+'_query';nodes=[];edges=[]
 for i,cid in enumerate(support):
  pid='PF_'+qid+'_'+str(i);nodes.append({'proofId':pid,'kind':'source','originalIds':[cid],'participantId':who,'propositionId':pids[0] if pids else None,'witnesses':witnesses(cid)});edges.append({'fromProofId':pid,'toProofId':root,'role':'supports'})
 if view:
  for i,b in enumerate([b for b in view['beliefs'] if b['propositionId'] in pids]):
   pid='PF_'+qid+'_belief_'+str(i);nodes.append({'proofId':pid,'kind':'endorsement','originalIds':[byact[a]['turnId'] for a in b['sourceActIds']],'participantId':who,'propositionId':b['propositionId'],'witnesses':clone(b['witnesses'])});edges.append({'fromProofId':pid,'toProofId':root,'role':'defeats' if b['status']=='defeated' else 'supports'})
 nodes.append({'proofId':root,'kind':'query','originalIds':[qid,target],'participantId':who,'propositionId':pids[0] if pids else None,'witnesses':witnesses(*support)})
 challenge={'passed':True,'unsupportedProofIds':[],'reason':'Authored source certificate preserves target, owner, actual adoption, participant access and original reported force.','witnesses':witnesses(*support)}
 observable={'performed':None,'truthValue':None,'beliefValue':None,'knowledgeValue':None,'due':None,'actorIds':[],'actKinds':[],'commitmentStatus':None,'referentIds':[]};observable.update(projection)
 graph={'nodes':nodes,'edges':edges,'rootProofIds':[root],'challenge':clone(challenge),'unresolved':[]}
 answers.append({'queryId':qid,'focus':focus,'observable':observable,'status':'refuted' if projection.get('truthValue') is False or projection.get('knowledgeValue') is False else 'supported','proof':graph,'challenge':challenge,'explanation':answer_explanations[qid]})
findings=[{'findingId':'LF_'+ident,'kind':ident,'passed':True,'originalIds':ids,'reason':reason,'witnesses':witnesses(*cids)} for ident,ids,reason,cids in [
('cross-view-adoption',['P17','W18','S18','T18_ADOPT'],'Last audience adoption changes issuer-owned normative state, with no invented issuer knowledge.',['P17','W18','S18','T18_ADOPT','R06']),
('terminal-independent-lineage',['P17','N18'],'Later true conditions preserve P17 terminal retraction; the fresh performed promise owns distinct audience/day/version predicates.',['P17','N18','R07','R09']),
('receipt-endorsement-separation',['RADOPT_O','O19_REPORT','O19_ADOPT'],'Verified arrival alone differs from explicit later endorsement.',['RADOPT_O','O19_REPORT','O19_ADOPT','R05']),
('forwarded-force',['REM19','W18'],'Reporting preserves original Oren withdrawal without a new Remy revision.',['REM19','W18','R03']),
('correction-without-adoption',['S19_EARLY','RC19_S'],'The source establishes actual correction arrival but no Sada adoption.',['S19_EARLY','RC19_S','R04'])]]
audits=[{'queryId':a['queryId'],'passed':True,'checkedProofIds':[n['proofId'] for n in a['proof']['nodes']],'reason':'Full typed source/historical witness for the declared projection; this is not NL execution.','witnesses':clone(a['focus']['witnesses'])} for a in answers]
report={'caseId':snapshot['caseId'],'source':source,'scoped':scoped,'episodePlan':plan,'evolution':evolution,'answers':answers,'findings':findings,'answerAudits':audits,'unresolved':[]}
(ROOT/'reference-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'scope':'One authored typed source witness, not application execution or teacher trajectory','turns':len(turns),'acts':len(acts),'episodes':len(episodes),'queries':len(answers)}))
