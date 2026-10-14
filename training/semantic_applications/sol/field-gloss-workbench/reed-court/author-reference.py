#!/usr/bin/env python3
"""Assemble ONE private authored LinguisticReport. This does not execute the NL app.

Semantic judgments below are explicit human-authored annotations for this fictional
world. Code does only Cartesian identity enumeration, source lookup/quotes, exact
surface offsets, data copying and prefix accounting. No inference/classifier is
exported to the executing application. The reference is not a teaching trajectory.
"""
from pathlib import Path
import copy, itertools, json, re
BASE=Path(__file__).resolve().parent
snapshot=json.loads((BASE/'snapshot.json').read_text())
request=json.loads((BASE/'request.json').read_text())
clauses=[]
for doc in snapshot['documents']:
    for line in doc['text'].splitlines():
        m=re.match(r'^\[([^\]]+)\]',line)
        if m: clauses.append(dict(clauseId=m[1],sourceId=doc['sourceId'],path=doc['path'],text=line))
by={c['clauseId']:c for c in clauses}
def ws(*ids):
    return [dict(clauseId=i,sourceId=by[i]['sourceId'],path=by[i]['path'],quote=by[i]['text'])for i in dict.fromkeys(ids)]
def prose(c):return by[c]['text'].split('] ',1)[1]
# Explicit authored original roles, not a parser of source meanings.
ROLE_IDS={
 'rule':['K01','K02','K03','K04','K05','K06','K07','K08','K09','O01','O02','O03','O04','O05','O06','A01','A02'],
 'candidate-domain':['D01','D02','D03','D04'],
 'context':['D05','C3','CTX1','CTX2','CTX3','T01'],
 'stem':['L01','L02','L03','L04'],'entity':['L01','L02','L03'],
 'session':['S1','S2','S3'],'situation':['SA1','SA2','SA3','SA4','SA5','SA6'],
 'prompt':['P1','P2','P3','P4','P5','P6'],
 'utterance':['U1','U2','U3','U4','U5','U6','X1','X2','X3'],
 'note':['J1','J2','J3','J4','J5','J6','N1','N2','N3','N1R'],
 'correction':['C1','C2'],'authorization':['O05']}
roles={i:[]for i in by}
for role,ids in ROLE_IDS.items():
    for i in ids:roles[i].append(role)
assert all(roles.values())
inventory=dict(entries=[dict(clause=c,actualReading=dict(clauseId=c['clauseId'],roles=roles[c['clauseId']],reason='Authored original source role.',witnesses=ws(c['clauseId']),unresolved=[]),status='indexed',failure=None)for c in clauses],unresolved=[])
# Original scene and target semantic annotations. Fields are not computed from language.
ROWS={
 'U1':('S1','SA1','P1',None,'emana ilara taku','Ila','Ema','Ema',True),
 'U2':('S1','SA2','P2',None,'emara ilana taku','Ema','Ila','Ila',True),
 'U3':('S2','SA3','P3',None,'emarana ila taku','Ema','Ila','Ema',True),
 'U4':('S3','SA4','P4',None,'narana emara taku','Ema','Nara','Nara',True),
 'U5':('S3','SA5','P5',None,'emara ilana takumi','Ema','Ila','Ila',False),
 'U6':('S3','SA6','P6',None,'emara narana taku','Ema','Nara','Nara',True),
 'X1':(None,None,None,'CTX1','narana emara takumi','Ema','Nara','Nara',False),
 'X2':(None,None,None,'CTX2','nararana ila taku','Nara','Ila','Nara',True),
 'X3':(None,None,None,'CTX3','ilara narana taku','Ila','Nara','Nara',True)}
situations=[];utterances=[]
for u,(session,situation,prompt,context,surface,agent,patient,topic,occurs)in ROWS.items():
    assert surface in by[u]['text']
    utterances.append(dict(utteranceId=u,sessionId=session,surface=surface,situationId=situation,promptId=prompt,contextId=context,sourceClauseId=u,witnesses=ws(u),unresolved=[]))
    if situation:
        situations.append(dict(situationId=situation,sessionId=session,entityIds=sorted([agent,patient]),agentId=agent,patientId=patient,eventId='LIFT',eventOccurs=occurs,topicId=topic,contrastSetIds=sorted([agent,patient]),temporalReading=None,witnesses=ws(situation,prompt),unresolved=[]))
sessions=[dict(sessionId=s,consultantId='Reni',promptIds=ps,situationIds=ss,contextClauseIds=ctx,occurredOrder=n,witnesses=ws(s,'A01'),unresolved=[])for s,ps,ss,ctx,n in [('S1',['P1','P2'],['SA1','SA2'],['A01','A02'],1),('S2',['P3'],['SA3'],['C1','N1R','C2','A01'],2),('S3',['P4','P5','P6'],['SA4','SA5','SA6'],['A01'],3)]]
notes=[dict(noteId='J'+str(n),utteranceId='U'+str(n),slotId=None,kind='consultant-judgment',meaning=prose('J'+str(n)),authorId='Reni',judgment='accepted',witnesses=ws('J'+str(n),'A01'),unresolved=[])for n in range(1,7)]
for id,u,slot,kind,author,judgment,support in [('N1','U1','NA_GLOSS','proposed-gloss','Belo',None,['N1']),('N2','U1','SURFACE_U1','transcription','Belo','accepted',['N2','C1']),('N3','U2','REMINDER','proposed-gloss','Belo',None,['N3']),('N1R','U1','NA_GLOSS','adopted-gloss','Belo',None,['N1R','C1','C2'])]:
    notes.append(dict(noteId=id,utteranceId=u,slotId=slot,kind=kind,meaning=prose(id),authorId=author,judgment=judgment,witnesses=ws(*support),unresolved=[]))
corrections=[dict(correctionId=c,targetNoteIds=['N1'],targetSlotIds=['NA_GLOSS'],replacementNoteIds=['N1R'],scope='gloss',adopted=True,excludedSlotIds=['SURFACE_U1','REMINDER','F_MI'],consultantId='Reni',reason='The selected correction and its explicit adoption govern only the original NA gloss.',witnesses=ws(c,'C1','C2','N1R','A01','A02'),unresolved=[])for c in ['C1','C2']]
slots=[dict(slotId=s,values=v,meaning=prose(c),witnesses=ws(c,'K04'))for s,v,c in [('B_NA',['productive','frozen'],'D01'),('F_NA',['agent','topic'],'D02'),('F_RA',['agent','patient'],'D03'),('F_MI',['event-negative','not-yet'],'D04')]]
stems=[dict(surface=s,entityId=e,eventId=v,meaning=prose(c),witnesses=ws(c))for s,e,v,c in [('ema','Ema',None,'L01'),('ila','Ila',None,'L02'),('nara','Nara',None,'L03'),('taku',None,'LIFT','L04')]]
# Private authored per-definition returns for the named readers, not runtime traces.
domain_entries=[dict(clause=by[c],actualReading=dict(clauseId=c,kind='domain',slots=[copy.deepcopy(slot)],reason='This original definition supplies this finite candidate slot.',witnesses=ws(c),unresolved=[]),status='read',failure=None)for c,slot in zip(ROLE_IDS['candidate-domain'],slots)]
stem_entries=[dict(clause=by[c],actualReading=dict(clauseId=c,kind='stem',stems=[copy.deepcopy(stem)],reason='This original definition supplies this source stem identity.',witnesses=ws(c),unresolved=[]),status='read',failure=None)for c,stem in zip(ROLE_IDS['stem'],stems)]
candidate_domains=dict(entries=domain_entries,slots=copy.deepcopy(slots),unresolved=[])
stem_definitions=dict(entries=stem_entries,stems=copy.deepcopy(stems),unresolved=[])
source=dict(caseId=snapshot['caseId'],clauses=clauses,inventory=inventory,rules=[by[c]for c in ROLE_IDS['rule']],sessions=dict(sessions=sessions,situations=situations,unresolved=[]),utteranceIndex=dict(utterances=utterances,notes=notes,unresolved=[]),corrections=dict(corrections=corrections,unresolved=[]),candidateDomains=candidate_domains,stemDefinitions=stem_definitions,candidateSlots=slots,stems=stems,entityIds=['Ema','Ila','Nara'],eventIds=['LIFT'],unresolved=[])
versions=[dict(originalNoteId=n['noteId'],effectiveNoteId='N1R'if n['noteId']=='N1'else n['noteId'],slotId=n['slotId'],status='replaced'if n['noteId']=='N1'else'retained',correctionIds=['C1','C2']if n['noteId']=='N1'else[],witnesses=ws(n['noteId'],*(['C1','C2','N1R']if n['noteId']=='N1'else[])),reason='Original scoped correction/adoption.'if n['noteId']=='N1'else'Untargeted original note retained.')for n in notes]
units=[];links=[]
for utterance in utterances:
    u=utterance['utteranceId'];r=ROWS[u];tested=r[0]is not None
    note_ids=(['J'+u[1:]]+(['N1R','N2']if u=='U1'else['N3']if u=='U2'else[]))if tested else[]
    selected=[n for n in notes if n['noteId']in note_ids]
    scope=dict(assertedFields=['agentId','patientId','eventId','eventOccurs','topicId']if tested else[],excludedFields=['futureExpected','temporalReading']if tested else[],judgmentNoteIds=['J'+u[1:]]if tested else[],status='usable'if tested else'unusable',reason='Actual prompt and consultant judgment constrain only the named fields.'if tested else'Source O04/T01 establishes this target as untested, not negative evidence.',witnesses=ws(*([r[2],'J'+u[1:],'K07']if tested else[u,'O04','T01'])),unresolved=[])
    alignment=dict(utteranceId=u,sessionId=r[0],situationId=r[1],promptId=r[2],scope=scope,effectiveNoteIds=note_ids,witnesses=ws(u,*([r[2]]if tested else[r[3]])),unresolved=[])
    units.append(dict(evidenceId='E_'+u,utterance=utterance,situation=next((s for s in situations if s['situationId']==r[1]),None),alignment=alignment,notes=selected,witnesses=ws(u),))
    for target,relation in [(r[1],'depicted-situation'),(r[2],'actual-prompt'),(r[3],'heldout-context')]:
        if target:links.append(dict(fromId=u,toId=target,relation=relation,witnesses=ws(u,target)))
    for n in note_ids:links.append(dict(fromId=n,toId=u,relation='effective-note-about-utterance',witnesses=ws(n,u)))
for c in corrections:
    links.append(dict(fromId=c['correctionId'],toId='N1',relation='targets-original-gloss',witnesses=c['witnesses']))
    links.append(dict(fromId=c['correctionId'],toId='N1R',relation='adopted-replacement',witnesses=c['witnesses']))
graph=dict(units=units,effectiveNotes=dict(entries=versions,unresolved=[]),links=links,unresolved=[])
# Explicit token annotations, separately authored for the two boundary analyses.
# Each tuple gives exact piece, original stem, function category, entity, event.
P={
 'ema':[('ema','ema',None,'Ema',None)],'ila':[('ila','ila',None,'Ila',None)],
 'emara':[('ema','ema',None,'Ema',None),('ra',None,'RA','Ema',None)],
 'ilara':[('ila','ila',None,'Ila',None),('ra',None,'RA','Ila',None)],
 'emana':[('ema','ema',None,'Ema',None),('na',None,'NA','Ema',None)],
 'ilana':[('ila','ila',None,'Ila',None),('na',None,'NA','Ila',None)],
 'emarana':[('ema','ema',None,'Ema',None),('ra',None,'RA','Ema',None),('na',None,'NA','Ema',None)],
 'narana':[('nara','nara',None,'Nara',None),('na',None,'NA','Nara',None)],
 'nararana':[('nara','nara',None,'Nara',None),('ra',None,'RA','Nara',None),('na',None,'NA','Nara',None)],
 'taku':[('taku','taku',None,None,'LIFT')],
 'takumi':[('taku','taku',None,None,'LIFT'),('mi',None,'MI',None,'LIFT')]}
F=copy.deepcopy(P)
for form,entity,category in [('emana','Ema','NA'),('ilana','Ila','NA'),('emarana','Ema','RA and NA')]:F[form]=[(form,form,category,entity,None)]
F['narana']=[('narana',None,None,None,None)]
# Authored prediction cells for U1 under the four role-choice families.
U1_PRED={
 ('agent','agent'):dict(licensed=False,agentId=None,patientId=None,topicId=None,contributions=[('Ema','agent','NA'),('Ila','agent','RA')],status='contradiction',reason='Two distinct entities fill the unique agent slot.'),
 ('agent','patient'):dict(licensed=True,agentId='Ema',patientId='Ila',topicId=None,contributions=[('Ema','agent','NA'),('Ila','patient','RA')],status='contradiction',reason='The predicted event roles are reversed from J1.'),
 ('topic','patient'):dict(licensed=True,agentId='Ema',patientId='Ila',topicId='Ema',contributions=[('Ema','topic','NA'),('Ila','patient','RA')],status='contradiction',reason='The predicted event roles are reversed from J1.'),
 ('topic','agent'):dict(licensed=True,agentId='Ila',patientId='Ema',topicId='Ema',contributions=[('Ema','topic','NA'),('Ila','agent','RA')],status='support',reason='Actual agent, patient, event and topic agree with J1.')}
# Every non-U1 attempted role cell is independently authored, not inferred from strings.
GOOD={
 'U2':dict(agentId='Ema',patientId='Ila',topicId='Ila',contributions=[('Ema','agent','RA'),('Ila','topic','NA')]),
 'U3':dict(agentId='Ema',patientId='Ila',topicId='Ema',contributions=[('Ema','agent','RA'),('Ema','topic','NA')]),
 'U4':dict(agentId='Ema',patientId='Nara',topicId='Nara',contributions=[('Nara','topic','NA'),('Ema','agent','RA')]),
 'U5':dict(agentId='Ema',patientId='Ila',topicId='Ila',contributions=[('Ema','agent','RA'),('Ila','topic','NA')]),
 'U6':dict(agentId='Ema',patientId='Nara',topicId='Nara',contributions=[('Ema','agent','RA'),('Nara','topic','NA')]),
 'X1':dict(agentId='Ema',patientId='Nara',topicId='Nara',contributions=[('Nara','topic','NA'),('Ema','agent','RA')]),
 'X2':dict(agentId='Nara',patientId='Ila',topicId='Nara',contributions=[('Nara','agent','RA'),('Nara','topic','NA')]),
 'X3':dict(agentId='Ila',patientId='Nara',topicId='Nara',contributions=[('Ila','agent','RA'),('Nara','topic','NA')])}
def segmentation(u,h,boundary):
    tokens=P if boundary=='productive'else F
    morphemes=[];decisions=[];cursor=0
    for token in ROWS[u][4].split(' '):
        start=cursor;parts=[]
        for piece,stem,function,entity,event in tokens[token]:
            end=cursor+len(piece)
            parts.append(dict(start=cursor,end=end,surface=piece,stem=stem,function=function,entityId=entity,eventId=event,witnesses=ws(u,'D01','K05',*(['L03']if token.startswith('nara')else[]))))
            cursor=end
        assert ''.join(p['surface']for p in parts)==token
        licensed=not(boundary=='frozen'and token=='narana')
        decisions.append(dict(start=start,end=cursor,segments=parts,licensed=licensed,reason='Authored candidate-specific boundary/licensing annotation.',quotes=ws(u,'D01','K05'),unresolved=[]))
        morphemes.extend(parts);cursor+=1
    return dict(utteranceId=u,hypothesisId=h,morphemes=morphemes,licensed=all(d['licensed']for d in decisions),boundaryDecisions=decisions,unresolved=[])
def actual_roles(u,h,seg,cell,future,licensed=True,unknown=[]):
    contributions=[]
    for entity,role,category in cell['contributions']:
        span=next(p for p in seg['morphemes']if p['entityId']==entity and category in (p['function']or'').split(' and '))
        contributions.append(dict(entityId=entity,role=role,start=span['start'],end=span['end'],witnesses=ws(u,'D02'if category=='NA'else'D03','K03')))
    meaning=dict(licensed=licensed,contributions=contributions,topicId=cell['topicId'],agentId=cell['agentId'],patientId=cell['patientId'],eventId='LIFT',eventOccurs=ROWS[u][8],temporalReading=None,futureExpected=future,reason='Authored candidate prediction retained independently from required scene meaning.',witnesses=ws(u,'D02','D03','D04','K01','K02','K03'),unresolved=list(unknown))
    return dict(utteranceId=u,hypothesisId=h,meaning=meaning,segmentation=copy.deepcopy(seg),unresolved=list(unknown))
# Human-authored rejection cutoffs for this fixed canonical order. Not a production invariant.
INITIAL=[];FINAL=[];attempts=[];skipped=[];rejections=[]
for n,values in enumerate(itertools.product(*[s['values']for s in slots]),1):
    h='H'+str(n).zfill(2);choice=dict(zip([s['slotId']for s in slots],values));initial=dict(hypothesisId=h,choices=[dict(slotId=k,value=v)for k,v in choice.items()],status='pending',reason='Original permitted choice vector awaiting actual evidence.',witnesses=ws('D01','D02','D03','D04'),)
    INITIAL.append(initial);final=copy.deepcopy(initial);prior_ids=[];reject=None
    # Explicit annotations attach to this originally enumerated vector.
    first=U1_PRED[(choice['F_NA'],choice['F_RA'])]
    cutoff='U1'if first['status']=='contradiction'else'U4'if choice['B_NA']=='frozen'else None
    for u in ['U1','U2','U3','U4','U5','U6']:
        if reject:
            skipped.append(dict(hypothesisId=h,evidenceId='E_'+u,rejectionAttemptId=reject));continue
        seg=segmentation(u,h,choice['B_NA']);future=True if u=='U5'and choice['F_MI']=='not-yet'else None
        if u=='U1':
            role=actual_roles(u,h,seg,first,future,first['licensed']);status=first['status'];reason=first['reason']
        elif u=='U4'and cutoff=='U4':
            cell=dict(agentId='Ema',patientId=None,topicId=None,contributions=[('Ema','agent','RA')]);role=actual_roles(u,h,seg,cell,future,None,['Frozen candidate has no licensed narana entry.']);status='contradiction';reason='Accepted U4 is outside the frozen candidate closed NA lexicon.'
        else:
            role=actual_roles(u,h,seg,GOOD[u],future);status='support';reason='Actual prediction agrees with the source-constrained fields; unasked future is not compared.'
        compared=['surfaceLicensed']if u=='U4'and cutoff=='U4'else['roleLicensing']if u=='U1'and not first['licensed']else['agentId','patientId','eventId','eventOccurs','topicId']
        comp=dict(status=status,comparedFields=compared,contradictedNoteIds=['J'+u[1:]]if status=='contradiction'else[],reason=reason,witnesses=ws(u,'J'+u[1:],'P'+u[1:],'D01'if u=='U4'and cutoff=='U4'else'K08'),unresolved=[])
        a='A_'+h+'_'+u;attempts.append(dict(attemptId=a,hypothesisId=h,evidenceId='E_'+u,incoming=copy.deepcopy(initial),segmentation=seg,roles=role,compatibility=comp))
        if status=='contradiction':reject=a;final['status']='rejected';final['reason']=reason;final['witnesses']=comp['witnesses'];rejections.append(dict(hypothesisId=h,firstContradictionAttemptId=a,earlierAttemptIds=list(prior_ids),unattemptedEvidenceIds=['E_'+v for v in ['U1','U2','U3','U4','U5','U6'][int(u[1:]):]],witnesses=comp['witnesses']))
        else:prior_ids.append(a)
    if not reject:final['status']='surviving';final['reason']='All six original usable comparisons completed; MI prospective distinction remains unasked.'
    FINAL.append(final)
survivors=[h['hypothesisId']for h in FINAL if h['status']=='surviving']
grammar=dict(inventory=dict(hypotheses=INITIAL,candidateSlotIds=[s['slotId']for s in slots],unresolved=[]),hypotheses=FINAL,survivorIds=survivors,attempts=attempts,rejections=rejections,skipped=skipped,unresolved=[])
translations=[];challenges=[]
for target in request['targets']:
    u=target['utteranceId'];readings=[]
    for h in survivors:
        hyp=next(r for r in FINAL if r['hypothesisId']==h);choice={r['slotId']:r['value']for r in hyp['choices']};future=True if u=='X1'and choice['F_MI']=='not-yet'else None
        seg=segmentation(u,h,'productive');roles_reading=actual_roles(u,h,seg,GOOD[u],future)
        ctx=dict(agentId=GOOD[u]['agentId'],patientId=GOOD[u]['patientId'],eventId='LIFT',eventOccurs=ROWS[u][8],topicId=GOOD[u]['topicId'],temporalReading=None,futureExpected=future,reason='Actual source-defined grammar contribution, not a claim that a future event occurs.',witnesses=ws(u,ROWS[u][3],'D04','O03','T01'),unresolved=[])
        readings.append(dict(readingId='R_'+target['targetId']+'_'+h,targetId=target['targetId'],utteranceId=u,hypothesisId=h,segmentation=seg,roles=roles_reading,context=ctx,witnesses=ws(u,ROWS[u][3]),unresolved=[]))
        challenges.append(dict(targetId=target['targetId'],hypothesisId=h,status='grounded',reason='The actual retained grammar/context contribution supports this reading; no held-out judgment is invented.',witnesses=ws(u,ROWS[u][3],'O04'),))
    groups=[dict(hypothesisIds=[r['hypothesisId']],readingIds=[r['readingId']],reason='Distinct prospective contribution retained.')for r in readings]if u=='X1'else[dict(hypothesisIds=survivors,readingIds=[r['readingId']for r in readings],reason='Actual completed positive meanings agree across the two surviving MI analyses.')]
    translations.append(dict(targetId=target['targetId'],readings=readings,equivalenceGroups=groups,ambiguityFields=['futureExpected']if u=='X1'else[],unresolved=[]))
doc=next(d for d in snapshot['documents']if d['path']==request['repair']['allowedPath']);start=doc['text'].index('NA_GLOSS: "')+len('NA_GLOSS: "');replacement='contrastive topic, which can be agent or patient'
meaning=dict(noteId='N1',slotId='NA_GLOSS',replacement=replacement,status='replace',supportedHypothesisIds=survivors,rejectedHypothesisIds=[h['hypothesisId']for h in FINAL if any(c['slotId']=='F_NA'and c['value']=='agent'for c in h['choices'])],reason='Actual consultant correction/adoption governs only this original proposed NA gloss; no MI claim is settled.',witnesses=ws('N1','C1','C2','N1R','O05','C3'),unresolved=[])
patch=dict(request=request['repair'],patches=[dict(path=doc['path'],start=start,end=start+5,before='agent',replacement=replacement,meaning=meaning)],unchangedNoteIds=[n['noteId']for n in notes if n['noteId']!='N1'],unresolved=[])
report=dict(caseId=snapshot['caseId'],source=source,graph=graph,grammar=grammar,translations=dict(targets=translations,unresolved=[]),patch=patch,challenges=challenges,sourceCoverage=[c['clauseId']for c in clauses],unresolved=[])
assert len(attempts)==32 and len(skipped)==64 and len(survivors)==2
(BASE/'reference-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(scope='One authored source certificate; no NL application execution',hypotheses=len(INITIAL),attempts=len(attempts),skips=len(skipped),survivors=survivors,readings=sum(len(t['readings'])for t in translations),original_clauses=len(clauses))))
