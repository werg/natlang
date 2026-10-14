"""Author one private typed certificate from explicit hand-authored semantic annotations.
This is exact JSON/provenance/state assembly, not an implementation or execution of the NL application.
It is excluded from both model-visible roots and cannot grant live/trajectory admission.
"""
from pathlib import Path
import copy,json,re
ROOT=Path(__file__).parent
snapshot=json.loads((ROOT/'snapshot.json').read_text());request=json.loads((ROOT/'request.json').read_text())
clauses={}
for doc in snapshot['documents']:
 for line in doc['text'].splitlines():
  match=re.match(r'^\[([^]]+)\]',line)
  if match:clauses[match[1]]={'clauseId':match[1],'sourceId':doc['sourceId'],'path':doc['path'],'text':line}
def witnesses(*ids):return [{**{k:clauses[i][k] for k in ['clauseId','sourceId','path']},'quote':clauses[i]['text']} for i in ids]
def moment(day,hour):return {'day':day,'hour':hour}
def tick(t):return 24*t['day']+t['hour']
def clone(x):return copy.deepcopy(x)
# Authored original utterance spans and metadata. None is inferred by a text classifier.
turn_annotations=[
('J09','Jon',9,16,['Mira','Lea'],'P','shared','I will collect the crate tomorrow.'),
('M10','Mira',10,9,['Lea','Jon'],'P','shared','If Jon collects the crate before day11 noon, I promise to release the key to Lea after Lea personally acknowledges shipping brief B2.'),
('L10','Lea',10,10,['Mira'],'P','shared','I personally acknowledge shipping brief B1, not B2.'),
('J11','Jon',11,10,['Lea'],'P','shared','I picked the crate up at10 today.'),
('L11','Lea',11,10.5,[],None,'private',"I believe Jon's claim that he collected before noon today. I have only his assertion, not verified collection evidence."),
('N11','Neri',11,11,[],None,'private','I doubt Jon picked it up. I do not adopt his collection claim.'),
('M12_PRIVATE','Mira',12,8,[],None,'private','Privately I want to cancel my conditional key promise. I have sent nobody a retraction.'),
('M12_REPORT','Mira',12,11,['Lea'],'P','shared','The verified collection log says Jon collected day12 hour9, not before day11 noon. Correct J11 on that point.'),
('M12_ARCHIVE','Mira',12,13,['Neri'],None,'shared','Jon said on day9 hour16, “I will collect the crate tomorrow.”'),
('N12','Neri',12,14,['Ada'],'R','shared',"Mira's authenticated archive summary M12_ARCHIVE from day12 hour13 was: «Jon said on day9 hour16, “I will collect the crate tomorrow.”» I am forwarding Mira's quotation of Jon, not promising collection myself."),
('A12','Ada',12,16,[],None,'private',"I believe Jon's collection promise is due day13 because I used Neri's forwarding day for tomorrow."),
('L13','Lea',13,10,[],None,'private',"I adopt Mira's correction and verified log. I retract my endorsement of J11: Jon did not collect before day11 noon; he collected day12 hour9."),
('N13','Neri',13,11,['Mira'],None,'shared','Great, another flawless handover.'),
('AMB13','Neri',13,11.5,['Ada'],None,'shared',"She said 'Excellent.'")]
turns=[]
for ident,actor,day,hour,audience,channel,visibility,content in turn_annotations:
 assert content in clauses[ident]['text'],ident
 turns.append({'turnId':ident,'clauseId':ident,'speaker':actor,'occurredAt':moment(day,hour),'audienceIds':audience,'channelId':channel,'visibility':visibility,'content':content,'witnesses':witnesses(ident)})
turn_by_id={t['turnId']:t for t in turns}
proposition_annotations={'P_DUE10':'Jon original collection promise due day10','P_DUE13':'collection promise due day13','P_COLLECT10':'Jon collected on day10','P_COLLECT11':'Jon collected before day11 noon','P_COLLECT12':'actual collection day12 hour9','P_ACK_B1':'Lea personally acknowledged B1','P_ACK_B2':'Lea personally acknowledged B2','P_RELEASE':'Mira released key to Lea','P_CANCEL_INTENT':'Mira privately intends to cancel M10','P_FLAWLESS':'handover was flawless'}
propositions=[{'propositionId':k,'meaning':v,'witnesses':witnesses('P01')} for k,v in proposition_annotations.items()]
physical_annotations=[('F_COLLECTION','P_COLLECT10',False,None,'F01'),('F_COLLECTION','P_COLLECT11',False,None,'F01'),('F_COLLECTION','P_COLLECT12',True,moment(12,9),'F01'),('F_NO_RELEASE','P_RELEASE',False,None,'F02'),('F_ACKNOWLEDGMENT','P_ACK_B1',True,moment(10,10),'F04'),('F_ACKNOWLEDGMENT','P_ACK_B2',False,None,'F04')]
physical=[{'factId':i,'propositionId':p,'value':v,'occurredAt':at,'completenessThrough':moment(13,12),'witnesses':witnesses(c)} for i,p,v,at,c in physical_annotations]
actors=['Mira','Lea','Jon','Neri','Ada']
receipt_annotations=[('H09',['J09'],'Mira',moment(9,16),False,[]),('H09',['J09'],'Lea',moment(9,16),False,[]),('H10',['M10'],'Lea',moment(10,9),False,[]),('H10',['M10'],'Jon',moment(10,9),False,[]),('H10',['L10'],'Mira',moment(10,10),False,[]),('H11',['J11'],'Lea',moment(11,10),False,[]),('H11',['J11'],'Neri',moment(11,10.75),False,[]),('R12_MIRA',[],'Mira',moment(12,10),True,['F_COLLECTION']),('R12_LEA',['M12_REPORT'],'Lea',None,True,['F_COLLECTION']),('R13_LEA',['M12_REPORT'],'Lea',moment(13,9),True,['F_COLLECTION']),('H_ARCHIVE',['M12_ARCHIVE','J09'],'Neri',moment(12,13),False,[]),('R13_ADA',['N12','M12_ARCHIVE','J09'],'Ada',moment(12,15),False,[]),('H13',['N13'],'Mira',moment(13,11),False,[]),('H13',['AMB13'],'Ada',moment(13,11.5),False,[])]
receipts=[{'receiptId':i,'turnIds':ts,'recipientId':p,'dispatchedAt':moment(12,11) if i in ['R12_LEA','R13_LEA'] else at,'receivedAt':at,'readAt':at,'verifiedEvidence':verified,'verifiedFactIds':facts,'witnesses':witnesses(i)} for i,ts,p,at,verified,facts in receipt_annotations]
source={'caseId':snapshot['caseId'],'clauses':list(clauses.values()),'rules':[c for k,c in clauses.items() if k.startswith('R') and k[1:].isdigit()], 'turns':turns,'propositions':propositions,'physicalFacts':physical,'temporal':{'entries':[{'turnId':t['turnId'],'originalTime':t['occurredAt'],'reportingTime':t['occurredAt'] if t['turnId'] in ['N12','M12_ARCHIVE'] else None,'reason':'Authenticated original metadata; quoted interiors retain their separately attributed earlier anchors.','witnesses':t['witnesses']} for t in turns],'unresolved':[]},'participants':{'participantIds':actors,'memberships':[{'participantId':a,'channelId':'P','from':moment(9,0),'through':moment(13,24),'witnesses':witnesses('C02')} for a in ['Mira','Lea','Jon']]+[{'participantId':a,'channelId':'R','from':None,'through':None,'witnesses':witnesses('C02')} for a in ['Neri','Ada']],'unresolved':[]},'receipts':{'receipts':receipts,'unresolved':[]},'unresolved':[]}
# Typed quote scopes are authored. Offsets are mechanically bound to those exact chosen interiors.
trees=[]
for turn in turns:
 tid=turn['turnId'];root={'span':{'spanId':'S_'+tid,'turnId':tid,'parentSpanId':None,'start':0,'end':len(turn['content']),'inheritedSpeaker':turn['speaker'],'inheritedTime':turn['occurredAt']},'speaker':turn['speaker'],'originalTime':turn['occurredAt'],'originalTurnId':tid,'mode':'own','childSpanIds':[],'witnesses':turn['witnesses'],'unresolved':[]};nodes=[root]
 children=[]
 if tid=='M12_ARCHIVE':children=[('J09',"I will collect the crate tomorrow.",root)]
 if tid=='N12':children=[('M12_ARCHIVE','Jon said on day9 hour16, “I will collect the crate tomorrow.”',root)]
 if tid=='AMB13':children=[(None,'Excellent.',root)]
 for original,content,parent in children:
  start=turn['content'].index(content);sid='S_'+tid+'_quoted';who=turn_by_id[original]['speaker'] if original else None;at=turn_by_id[original]['occurredAt'] if original else None
  child={'span':{'spanId':sid,'turnId':tid,'parentSpanId':parent['span']['spanId'],'start':start,'end':start+len(content),'inheritedSpeaker':who,'inheritedTime':at},'speaker':who,'originalTime':at,'originalTurnId':original,'mode':'quoted','childSpanIds':[],'witnesses':turn['witnesses'],'unresolved':['Speaker and original time are not disambiguated.'] if original is None else []};parent['childSpanIds'].append(sid);nodes.append(child)
  if tid=='N12':
   text='I will collect the crate tomorrow.';start=turn['content'].index(text);sid='S_N12_inner';child['childSpanIds'].append(sid);nodes.append({'span':{'spanId':sid,'turnId':tid,'parentSpanId':child['span']['spanId'],'start':start,'end':start+len(text),'inheritedSpeaker':'Jon','inheritedTime':moment(9,16)},'speaker':'Jon','originalTime':moment(9,16),'originalTurnId':'J09','mode':'quoted','childSpanIds':[],'witnesses':turn['witnesses'],'unresolved':[]})
 trees.append({'turnId':tid,'nodes':nodes,'unresolved':['Original Excellent attribution/force remains ambiguous.'] if tid=='AMB13' else []})
tree_by_id={t['turnId']:t for t in trees}
def predicate(pid,role,prop,cutoff=None,strict=False,subject=None,obj=None,clauses_used=('M10',)):
 return {'predicateId':pid,'role':role,'propositionId':prop,'cutoff':cutoff,'strictBefore':strict,'subjectId':subject,'objectId':obj,'reason':'Source-defined '+role+' predicate; actor, version and temporal scope are separate.','witnesses':witnesses(*clauses_used)}
collect_condition=predicate('C_BEFORE_NOON','activation','P_COLLECT11',moment(11,12),True,'Jon','crate')
ack_condition=predicate('C_PERSONAL_B2','activation','P_ACK_B2',None,False,'Lea','B2')
release_consequent=predicate('F_RELEASE','fulfillment','P_RELEASE',None,False,'Mira','key')
collection_consequent=predicate('F_COLLECT_DAY10','fulfillment','P_COLLECT10',moment(10,None),False,'Jon','crate',('J09','R11'))
# These meanings are an authored certificate, not a rule implementation or model input.
meanings={
'J09':('promise',True,[],['P_DUE10','P_COLLECT10'],'Jon',['Mira','Lea'],moment(10,None),[],[collection_consequent],[],False),
'M10':('promise',True,[],['P_COLLECT11','P_ACK_B2','P_RELEASE'],'Mira',['Lea'],None,[collect_condition,ack_condition],[release_consequent],[],False),
'L10':('acknowledgment',True,[('P_ACK_B1',True),('P_ACK_B2',False)],['P_ACK_B1','P_ACK_B2'],None,[],None,[],[],[],True),
'J11':('assertion',True,[('P_COLLECT11',True)],['P_COLLECT11'],None,[],None,[],[],[],True),
'L11':('assertion',True,[('P_COLLECT11',True)],['P_COLLECT11'],None,[],None,[],[],[],True),
'N11':('doubt',True,[('P_COLLECT11',None)],['P_COLLECT11'],None,[],None,[],[],[],False),
'M12_PRIVATE':('intention',True,[('P_CANCEL_INTENT',True)],['P_CANCEL_INTENT'],None,[],None,[],[],[],True),
'M12_REPORT':('correction',True,[('P_COLLECT11',False),('P_COLLECT12',True)],['P_COLLECT11','P_COLLECT12'],None,[],None,[],[],['J11'],True),
'M12_ARCHIVE':('report',True,[],['P_DUE10'],None,[],None,[],[],[],False),
'N12':('report',True,[],['P_DUE10'],None,[],None,[],[],[],False),
'A12':('assertion',True,[('P_DUE13',True)],['P_DUE13'],None,[],None,[],[],[],True),
'L13':('correction',True,[('P_COLLECT11',False),('P_COLLECT12',True)],['P_COLLECT11','P_COLLECT12'],None,[],None,[],[],['J11'],True),
'N13':('irony',True,[],['P_FLAWLESS'],None,[],None,[],[],[],False),
'AMB13':('report',True,[],[],None,[],None,[],[],[],False)}
acts=[]
for turn in turns:
 for index,node in enumerate(tree_by_id[turn['turnId']]['nodes']):
  tid=turn['turnId'];key=tid if index==0 else node['originalTurnId'];spec=meanings[key] if key else ('unresolved',False,[],[],None,[],None,[],[],[],None)
  kind,performed,claims,props,issuer,beneficiaries,due,conditions,consequences,supersedes,endorsement=spec
  if index:performed=False;endorsement=False;claims=[]
  meaning={'performed':performed,'kind':kind,'propositionIds':props,'claims':[{'propositionId':p,'value':v} for p,v in claims],'issuerId':issuer,'beneficiaryIds':beneficiaries,'due':due,'conditions':clone(conditions),'consequences':clone(consequences),'supersedesTurnIds':supersedes,'endorsement':endorsement,'alternatives':['sincere assertion','ironic praise'] if key is None else [],'reason':'Actual authored speech force in its original scope; quotation preserves meaning without performing a new promise.','quotes':[turn['content'][node['span']['start']:node['span']['end']]]}
  acts.append({'actId':'A_'+tid+'_'+str(index),'turnId':tid,'spanId':node['span']['spanId'],'originalTurnId':node['originalTurnId'],'speaker':node['speaker'],'occurredAt':node['originalTime'],'audienceIds':turn['audienceIds'] if index==0 else [],'meaning':meaning,'witnesses':turn['witnesses']})
act_by_id={a['actId']:a for a in acts};primary={t['turnId']:next(a for a in acts if a['turnId']==t['turnId']) for t in turns}
references=[]
for turn in turns:
 bindings=[];tid=turn['turnId']
 for node in tree_by_id[tid]['nodes']:
  text=turn['content'][node['span']['start']:node['span']['end']]
  if node['originalTurnId']=='J09':
   m={'expression':'tomorrow','kind':'time','referentIds':[],'anchor':moment(9,16),'dayOffset':1,'resolvedAt':moment(10,None),'status':'resolved','reason':'Original Jon utterance anchor, not forwarding day.','quotes':['tomorrow']};bindings.append({'spanId':node['span']['spanId'],'meaning':m,'witnesses':turn['witnesses']})
 if tid=='A12':bindings.append({'spanId':'S_A12','meaning':{'expression':'tomorrow','kind':'time','referentIds':[],'anchor':moment(12,14),'dayOffset':1,'resolvedAt':moment(13,None),'status':'resolved','reason':'Ada explicitly uses the wrong forwarding-day anchor in her endorsed private belief; the original promise remains anchored day9.','quotes':['tomorrow']},'witnesses':turn['witnesses']})
 if tid=='AMB13':bindings.append({'spanId':'S_AMB13','meaning':{'expression':'She','kind':'person','referentIds':['Mira','Ada'],'anchor':None,'dayOffset':None,'resolvedAt':None,'status':'ambiguous','reason':'Both source-eligible referents remain; no disambiguating intention.','quotes':['She']},'witnesses':turn['witnesses']})
 references.append({'turnId':tid,'bindings':bindings,'unresolved':['Pronoun attribution is source-explicit ambiguity.'] if tid=='AMB13' else []})
scope_findings=[{'findingId':'SC_'+t['turnId'],'kind':'scope-consistency','passed':True,'originalIds':[t['turnId']],'reason':'Actual scope is preserved, including source-declared false perspective or ambiguity where applicable.','witnesses':t['witnesses']} for t in turns]
scoped={'quoteTrees':trees,'references':references,'acts':acts,'scopeFindings':scope_findings,'unresolved':['AMB13 speaker and Excellent force remain source-explicit ambiguity.']}
# Source-established episode units; sorting and state copying are exact assembly mechanisms.
episode_units={}
def event(at):
 key=(at['day'],at['hour']);return episode_units.setdefault(key,{'episodeId':'E_'+str(at['day'])+'_'+str(at['hour']).replace('.','p'),'at':at,'kind':'utterance','turnIds':[],'actIds':[],'factIds':[],'availability':[],'witnesses':[]})
def availability(ep,actor,turn_ids,receipt_ids,fact_ids,clause_ids):
 ids=[a['actId'] for a in acts if a['turnId'] in turn_ids]
 row={'participantId':actor,'turnIds':turn_ids,'actIds':ids,'receiptIds':receipt_ids,'verifiedFactIds':fact_ids,'availableAt':ep['at'],'reason':'Original speaker ownership or actual named hearing/arrival; availability alone is not endorsement.','witnesses':witnesses(*clause_ids)}
 ep['availability'].append(row);ep['witnesses']+=row['witnesses'];return row
for t in turns:
 ep=event(t['occurredAt']);ep['turnIds'].append(t['turnId']);ep['actIds'].append(primary[t['turnId']]['actId']);ep['witnesses']+=t['witnesses'];availability(ep,t['speaker'],[t['turnId']],[],[],[t['turnId']])
for r in receipts:
 if r['receivedAt'] is None:continue
 ep=event(r['receivedAt']);ep['kind']='receipt' if not ep['turnIds'] else ep['kind'];availability(ep,r['recipientId'],r['turnIds'],[r['receiptId']],r['verifiedFactIds'],[r['receiptId']])
ep=event(moment(12,9));ep['kind']='physical';ep['factIds']=['F_COLLECTION'];ep['witnesses']+=witnesses('F01')
ep=event(moment(13,12));ep['kind']='physical';ep['factIds']=['F_NO_RELEASE','F_ACKNOWLEDGMENT'];ep['witnesses']+=witnesses('F02','F04')
for at,rule in [(moment(11,0),'R11'),(moment(11,12),'R07')]:
 ep=event(at);ep['kind']='boundary';ep['witnesses']+=witnesses(rule)
episodes=[episode_units[k] for k in sorted(episode_units)];plan={'episodes':episodes,'unresolved':[]}
views=[{'participantId':p,'availableTurnIds':[],'availableActIds':[],'availableVerifiedFactIds':[],'beliefs':[],'commitments':[],'revisionIds':[],'adoptedRevisionActIds':[]} for p in actors]
view_by_id={v['participantId']:v for v in views};history=[];deltas=[]
# Hand-authored endorsement journal. Fields are actual interpreted conclusions, not a classifier.
endorsement_annotations={
'J11':{'posture':'endorse','basis':['J11'],'adopts':[],'defeats':[]},
'L10':{'posture':'endorse','basis':['L10'],'adopts':[],'defeats':[]},
'L11':{'posture':'endorse','basis':['J11'],'adopts':[],'defeats':[]},
'N11':{'posture':'doubt','basis':['J11'],'adopts':[],'defeats':[]},
'M12_PRIVATE':{'posture':'endorse','basis':['M12_PRIVATE'],'adopts':[],'defeats':[]},
'M12_REPORT':{'posture':'endorse','basis':['M12_REPORT'],'adopts':[],'defeats':[]},
'A12':{'posture':'endorse','basis':['A12'],'adopts':[],'defeats':[]},
'L13':{'posture':'endorse','basis':['M12_REPORT'],'adopts':['M12_REPORT'],'defeats':['J11']}}
def condition_reading(pred,ep):
 # Actual values below are authored temporal annotations for the source predicates.
 now=tick(ep['at']);pid=pred['predicateId'];actual=None
 if pid=='C_BEFORE_NOON':actual=False if now>=tick(moment(11,12)) else None
 if pid=='C_PERSONAL_B2':actual=False
 if pid=='F_RELEASE':actual=False
 if pid=='F_COLLECT_DAY10':actual=False if ep['at']['day']>10 else None
 issuer='Jon' if pid=='F_COLLECT_DAY10' else 'Mira';v=view_by_id[issuer]
 matching=[b for b in v['beliefs'] if b['status']=='endorsed' and b['propositionId']==pred['propositionId'] and b['value']==actual]
 known=actual is not None and any(b['verifiedFactIds'] for b in matching)
 bview=view_by_id['Lea'];belief=[b for b in bview['beliefs'] if b['status']=='endorsed' and b['propositionId']==pred['propositionId']]
 perceived=belief[-1]['value'] if belief else None
 return {'predicateId':pid,'actual':actual,'issuerKnows':known,'beneficiaryBelieves':perceived,'reason':'Source-resolved pending window or bounded actual event truth; knowledge tests corroboration of that truth, belief retains its own value.','witnesses':pred['witnesses']+witnesses('R05','R07')}
for ep in episodes:
 ed={'episodeId':ep['episodeId'],'endorsements':[],'beliefs':[],'commitments':[],'revisions':[],'availability':clone(ep['availability'])}
 for row in ep['availability']:
  v=view_by_id[row['participantId']]
  for key,items in [('availableTurnIds',row['turnIds']),('availableActIds',row['actIds']),('availableVerifiedFactIds',row['verifiedFactIds'])]:
   v[key]=list(dict.fromkeys(v[key]+items))
 # Original utterance conclusions are applied from the authored journal. Receipt-only rows produce no beliefs.
 for tid in ep['turnIds']:
  act=primary[tid];actor=act['speaker'];view=view_by_id[actor];annotation=endorsement_annotations.get(tid)
  if annotation:
   claims=clone(act['meaning']['claims']);basis=[primary[i]['actId'] for i in annotation['basis']];adopts=[primary[i]['actId'] for i in annotation['adopts']];defeats=[primary[i]['actId'] for i in annotation['defeats']]
   decision={'participantId':actor,'actId':act['actId'],'claims':claims,'posture':annotation['posture'],'basisActIds':basis,'defeatsActIds':defeats,'adoptsActIds':adopts,'reason':'Authored explicit actor endorsement/doubt with original accessible basis and scope.','quotes':[turn_by_id[tid]['content']]};ed['endorsements'].append(decision)
   bd={'participantId':actor,'add':[],'defeatBeliefIds':[],'unresolved':[]}
   for old in view['beliefs']:
    if old['status']=='endorsed' and set(old['sourceActIds'])&set(defeats):bd['defeatBeliefIds'].append(old['beliefId']);old['status']='defeated'
   for index,claim in enumerate(claims):
    corroboration=[f['factId'] for f in physical if f['propositionId']==claim['propositionId'] and f['value']==claim['value'] and f['factId'] in view['availableVerifiedFactIds']]
    atom={'beliefId':'B_'+tid+'_'+str(index),'participantId':actor,'propositionId':claim['propositionId'],'value':claim['value'],'status':'doubted' if annotation['posture']=='doubt' else 'endorsed','acquiredAt':ep['at'],'sourceActIds':list(dict.fromkeys([act['actId']]+basis)),'availabilityEpisodeIds':[ep['episodeId']],'verifiedFactIds':list(dict.fromkeys(corroboration)),'witnesses':act['witnesses']};bd['add'].append(clone(atom));view['beliefs'].append(atom)
   ed['beliefs'].append(bd);view['adoptedRevisionActIds']=list(dict.fromkeys(view['adoptedRevisionActIds']+adopts))
   if tid=='L13':
    rev={'selection':{'revisionActId':primary['M12_REPORT']['actId'],'targetActIds':[primary['J11']['actId']],'targetPropositionIds':['P_COLLECT11','P_COLLECT12'],'kind':'belief-correction','communicated':True,'reason':'Lea explicitly adopts the original communicated M12_REPORT correction, targeting J11 collection-time claims without canceling its promise.','witnesses':witnesses('L13','M12_REPORT','R06'),'unresolved':[]},'revisionId':'REV_L13','participantId':actor,'commitmentOwnerId':None,'adoptingParticipantIds':[actor],'supersededActIds':defeats,'defeatBeliefIds':bd['defeatBeliefIds'],'retractCommitmentIds':[],'reason':'Lea explicitly adopts correction of J11; only her old endorsement is defeated, not another actor or an original promise.','witnesses':witnesses('L13','R13_LEA','R06'),'unresolved':[]};ed['revisions'].append(rev);view['revisionIds'].append(rev['revisionId'])
  else:
   posture='report-only' if act['meaning']['kind'] in ['report','irony'] else 'none'
   ed['endorsements'].append({'participantId':actor,'actId':act['actId'],'claims':[],'posture':posture,'basisActIds':[],'defeatsActIds':[],'adoptsActIds':[],'reason':'Speech force supplies no explicit belief adoption under the closed model.','quotes':[turn_by_id[tid]['content']]})
  if tid in ['J09','M10']:
   v=view_by_id[actor];cm={'commitmentId':'K_'+tid,'issuerId':actor,'beneficiaryIds':act['meaning']['beneficiaryIds'],'sourceActId':act['actId'],'propositionIds':[p['propositionId'] for p in act['meaning']['consequences']],'due':act['meaning']['due'],'conditions':[],'fulfillment':[],'status':'active' if tid=='J09' else 'conditional','establishedAt':ep['at'],'witnesses':act['witnesses']};v['commitments'].append(cm)
   ed['commitments'].append({'participantId':actor,'add':[clone(cm)],'replace':[],'unresolved':[]})
 # Exact construction of the actual recorded commitment evidence from the authored predicate annotations.
 for v in views:
  replacements=[]
  for cm in v['commitments']:
   act=act_by_id[cm['sourceActId']];cm['conditions']=[condition_reading(p,ep) for p in act['meaning']['conditions']];cm['fulfillment']=[condition_reading(p,ep) for p in act['meaning']['consequences']]
   if cm['sourceActId']==primary['J09']['actId'] and ep['at']['day']>10:cm['status']='failed'
   replacements.append(clone(cm))
  if replacements:ed['commitments'].append({'participantId':v['participantId'],'add':[],'replace':replacements,'unresolved':[]})
 history.append({'episodeId':ep['episodeId'],'at':ep['at'],'views':clone(views)});deltas.append(ed)
evolution={'views':clone(views),'history':history,'deltas':deltas,'unresolved':[]}
# These answer annotations are independent authored source conclusions, not an app implementation.
# Generated graph identities and JSON copying are exact private certificate assembly.
answer_annotations={
'Q01':(['J09','N12','R03','R11'],['P_DUE10'],['J09'],{'actorIds':['Jon'],'due':moment(10,None),'performed':True},'supported','Original Jon promise is anchored at day9; the inner tomorrow is day10, with no stated hour.'),
'Q02':(['A12','R04','R03'],['P_DUE13'],['A12'],{'beliefValue':True},'supported','Ada explicitly endorses day13 by her mistaken forwarding-day anchor; endorsed belief can be false.'),
'Q03':(['L11','H11','R04','R12_LEA'],['P_COLLECT11'],['L11','J11'],{'beliefValue':True},'supported','Lea old endorsement remains before actual receipt and adoption of the delayed correction.'),
'Q04':(['L11','H11','R05'],['P_COLLECT11'],['L11','J11'],{'knowledgeValue':False},'refuted','The endorsed collection claim lacks available verified corroboration; belief is not qualifying knowledge.'),
'Q05':(['L11','R13_LEA','R04','R06'],['P_COLLECT11'],['L11','J11','M12_REPORT'],{'beliefValue':True},'supported','Receipt supplies evidence availability, but only the later explicit adoption defeats the old endorsement.'),
'Q06':(['L13','R13_LEA','R06'],['P_COLLECT11'],['L13','M12_REPORT'],{'beliefValue':False},'supported','Lea adopts the false-before-noon correction and defeats her own earlier J11 endorsement.'),
'Q07':(['L13','R13_LEA','F01','R05'],['P_COLLECT12'],['L13','M12_REPORT'],{'knowledgeValue':True},'supported','The day12 collection proposition is endorsed with actually available verified log evidence.'),
'Q08':(['M10','L10','F01','F04','M12_PRIVATE','R06','R07','R08'],['P_COLLECT11','P_ACK_B2','P_RELEASE'],['M10'],{'actorIds':['Mira'],'commitmentStatus':'conditional'},'supported','Actual activation predicates are false; private cancellation intention communicates no adopted retraction.'),
'Q09':(['F02','R07'],['P_RELEASE'],[],{'truthValue':False},'refuted','The complete physical log through the query horizon establishes no key release.'),
'Q10':(['N11','H11','R04'],['P_COLLECT11'],['N11','J11'],{'beliefValue':None},'supported','Neri explicitly doubts and does not adopt; known absence of endorsement is not endorsement of falsehood.'),
'Q11':(['N13','R09'],['P_FLAWLESS'],['N13'],{'actKinds':['irony'],'performed':True},'supported','Signed tone framing makes the performed remark an ironic complaint, without literal praise endorsement.'),
'Q12':(['AMB13','R09'],[],['AMB13'],{'referentIds':['Mira','Ada']},'ambiguous','Both source-eligible female referents survive; the intention is not supplied.'),
'Q13':(['AMB13','R09'],[],['AMB13'],{},'ambiguous','The quoted Excellent retains sincere assertion and ironic praise alternatives; no decisive tone frame exists.'),
'Q14':(['N12','M12_ARCHIVE','J09','R03'],['P_DUE10'],['N12','M12_ARCHIVE'],{'performed':False},'supported','The reporters perform reports, not a new collection promise; embedded Jon force is depicted in its original scope.'),
'Q15':(['R12_MIRA','R05','R04'],['P_COLLECT11'],[],{'knowledgeValue':False},'refuted','Mira has received verified evidence but has not yet endorsed the corrected proposition.'),
'Q16':(['L10','F04','R08'],['P_ACK_B2'],['L10'],{'truthValue':False},'refuted','The only personal acknowledgment names B1; the complete version-specific log establishes no B2 acknowledgment.')}
answers=[]
for query in request['queries']:
 qid=query['queryId'];support,props,act_turns,projection,status,explanation=answer_annotations[qid]
 # Explicit original annotations are checked against source; this is not a semantic reader.
 for cid in support:assert cid in clauses,(qid,cid)
 actor=query['perspective'];eligible=[h for h in history if tick(h['at'])<=tick(query['asOf'])]
 view=next(v for v in eligible[-1]['views'] if v['participantId']==actor) if actor else None
 focus={'query':clone(query),'participantId':actor,'asOf':clone(query['asOf']),'propositionIds':props,'actIds':[primary[t]['actId'] for t in act_turns],'view':clone(view),'kind':query['dimension'],'reason':'Actual requested source lineage and at-or-before participant view; other perspectives are not substituted.','witnesses':witnesses(*support),'unresolved':[]}
 nodes=[];edges=[];root='PF_'+qid+'_query'
 for index,cid in enumerate(support):
  pid='PF_'+qid+'_'+str(index);nodes.append({'proofId':pid,'kind':'source','originalIds':[cid],'participantId':actor,'propositionId':props[0] if props else None,'witnesses':witnesses(cid)});edges.append({'fromProofId':pid,'toProofId':root,'role':'supports'})
 if actor and view:
  related=[b for b in view['beliefs'] if b['propositionId'] in props]
  for index,b in enumerate(related):
   pid='PF_'+qid+'_belief_'+str(index);nodes.append({'proofId':pid,'kind':'endorsement','originalIds':[act_by_id[a]['turnId'] for a in b['sourceActIds']],'participantId':actor,'propositionId':b['propositionId'],'witnesses':clone(b['witnesses'])});edges.append({'fromProofId':pid,'toProofId':root,'role':'defeats' if b['status']=='defeated' else 'supports'})
 nodes.append({'proofId':root,'kind':'query','originalIds':[qid,query['targetId']],'participantId':actor,'propositionId':props[0] if props else None,'witnesses':witnesses(*support)})
 challenge={'passed':True,'unsupportedProofIds':[],'reason':'Authored source witness separates original force, actual availability and endorsed perspective; source-declared ambiguity is retained.','witnesses':witnesses(*support)}
 observable={'performed':None,'truthValue':None,'beliefValue':None,'knowledgeValue':None,'due':None,'actorIds':[],'actKinds':[],'commitmentStatus':None,'referentIds':[]};observable.update(projection)
 graph={'nodes':nodes,'edges':edges,'rootProofIds':[root],'challenge':clone(challenge),'unresolved':[]}
 answers.append({'queryId':qid,'focus':focus,'observable':observable,'status':status,'proof':graph,'challenge':challenge,'explanation':explanation})
findings=[{'findingId':'LF_'+kind,'kind':kind,'passed':True,'originalIds':ids,'reason':reason,'witnesses':witnesses(*ws)} for kind,ids,reason,ws in [
('quote-scope',['J09','M12_ARCHIVE','N12'],'Nested forwarding preserves original Jon day9 scope, without a new promise.',['J09','M12_ARCHIVE','N12','R03']),
('availability-adoption',['M12_REPORT','R13_LEA','L13'],'Actual arrival and later explicit adoption are separate episode transitions.',['M12_REPORT','R12_LEA','R13_LEA','L13','R06']),
('activation-fulfillment',['M10','P_COLLECT11','P_ACK_B2','P_RELEASE'],'Actual antecedent, perspective evidence and consequent fulfillment remain separate typed values.',['M10','F01','F02','F04','R07','R08']),
('private-intention',['M12_PRIVATE','M10'],'Private intention is not a communicated adopted promise retraction.',['M12_PRIVATE','R06']),
('source-ambiguity',['AMB13'],'Missing attribution/tone is retained as source ambiguity, not missing execution progress.',['AMB13','R09'])]]
audits=[{'queryId':a['queryId'],'passed':True,'checkedProofIds':[n['proofId'] for n in a['proof']['nodes']],'reason':'Typed original source and historical view witness supports this projection; semantic execution remains independently unmeasured.','witnesses':clone(a['focus']['witnesses'])} for a in answers]
report={'caseId':snapshot['caseId'],'source':source,'scoped':scoped,'episodePlan':plan,'evolution':evolution,'answers':answers,'findings':findings,'answerAudits':audits,'unresolved':['AMB13 attribution and Excellent force remain genuinely source-ambiguous.']}
(ROOT/'reference-report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'scope':'One authored source/type certificate; not NL execution or teacher trajectory.','turns':len(turns),'clauses':len(clauses),'acts':len(acts),'episodes':len(episodes),'queries':len(answers)}))
