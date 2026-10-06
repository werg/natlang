// Source-backed bounded decisions over exact, reviewed HotpotQA training records.
// This family is a source-generation candidate only; it does not confer trajectory or training admission.
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {Random,curriculumCase,evalCall,nlFile,returnCall} from './lib.mjs';

const FIXTURE_BYTES=readFileSync(new URL('./decision-source-cards-fixtures.json',import.meta.url));
const FIXTURE=JSON.parse(FIXTURE_BYTES.toString('utf8'));
const sha=value=>createHash('sha256').update(value).digest('hex');
const HELD=new Set((FIXTURE.held_proposals??[]).map(row=>row.proposal_id));
const ROWS=FIXTURE.records.filter(row=>!HELD.has(row.proposal_id));
if(ROWS.length<2) throw new Error('decision_source_cards requires at least two active reviewed source facts');
const TYPE_SOURCE=`export type SourceCard = { card_id: string; title: string; sentences: string[] };
export type AnswerCandidate = { candidate_id: string; answer: string; source_case_id: string; source_card_ids: string[] };
export type SourceTask = { ordinal: number; proposal_id: string; source_case_id: string; primary_raw_record_id: string; question: string; sourceCards: SourceCard[]; candidateRecords: AnswerCandidate[] };
`;

function makeTask(row,ordinal,rng){
  const ordered=rng.shuffle(row.choices.map((answer,index)=>({answer,index})));
  const cards=row.source_cards.map(card=>({card_id:card.card_id,title:card.title,sentences:card.sentences}));
  const candidateRecords=ordered.map(({answer,index},candidateIndex)=>({
    candidate_id:`${row.proposal_id}:choice-${candidateIndex}`,
    answer,
    source_case_id:row.source_case_id,
    source_card_ids:cards.map(card=>card.card_id),
    _fixture_choice_index:index,
  }));
  const expected=candidateRecords.find(candidate=>candidate._fixture_choice_index===row.gold_choice_index);
  if(!expected) throw new Error(`missing reviewed gold for ${row.proposal_id}`);
  // The grading record is a closed structured source choice; the private shuffle index is never returned.
  const candidates=candidateRecords.map(({_fixture_choice_index,...candidate})=>candidate);
  const selected=candidates.find(candidate=>candidate.candidate_id===expected.candidate_id);
  return {
    task:{ordinal,proposal_id:row.proposal_id,source_case_id:row.source_case_id,
      primary_raw_record_id:row.primary_raw_record_id,question:row.question,sourceCards:cards,candidateRecords:candidates},
    selected,
  };
}

export function decisionSourceCards(seed,index,split='train'){
  if(split!=='train') throw new Error('decision_source_cards uses HotpotQA train-only source rows; test/held-out generation is not supported');
  // Each shape joins two reviewed facts. Wrap the odd final pair without
  // claiming repeated source facts are independent; held rows are never used.
  const pair=index%Math.ceil(ROWS.length/2), start=pair*2;
  const selectedRows=[ROWS[start],ROWS[(start+1)%ROWS.length]];
  if(selectedRows.length!==2) throw new Error(`incomplete reviewed source pair ${pair}`);
  const rng=new Random(seed,`decision-source-cards:${index}:${pair}`);
  const items=selectedRows.map((row,offset)=>makeTask(row,offset,rng));
  const batch=items.map(item=>item.task);
  const expected=items.map(item=>item.selected);
  const inlineQuestion='Choose the exact candidate answer record supported by all titled source cards for the runtime question. Read each card sentence and resolve cross-card references as needed. Return the candidate unchanged.';
  const selectCode=`const question = "Which candidate answer is supported by these source cards? "+task.question;
const sourceCards = task.sourceCards;
const candidateRecords = task.candidateRecords;
const selected = await nl<AnswerCandidate>\`\${question} Read every titled source card in sourceCards and compare its sentences with the candidate answer records in candidateRecords. Resolve references between cards when the question requires them. Return the exact supported candidate record unchanged.\`();
return selected;`;
  const rootCode=`const indexed = batch.map((task, ordinal) => ({task, ordinal}));
const resolved = await Promise.all(indexed.map(item => select_answer(item.task)));
return resolved;`;
  const children=[];
  for(const item of items){
    children.push({match:['You are inside this call: select_answer(',item.task.proposal_id],calls:[
      evalCall(selectCode),returnCall(item.selected),
    ]});
    children.push({match:['You are inside this call: nl@eval:',item.task.proposal_id],calls:[
      evalCall('return {question, sourceCards, candidateRecords};'),returnCall(item.selected),
    ]});
  }
  const candidateFiles=Object.fromEntries(items.map(item=>{
    const id=item.task.proposal_id;
    return [`criteria/${id}.json`,JSON.stringify({question:item.task.question,sourceCards:item.task.sourceCards,candidateRecords:item.task.candidateRecords},null,2)+'\n'];
  }));
  const files={
    'types.ts':TYPE_SOURCE,
    'answer_source_batch/select_answer.nl':nlFile({args:{task:'SourceTask'},returns:'AnswerCandidate',instructions:'Formulate the inline question from task.question at runtime. Read every titled source card in task.sourceCards and compare its sentences with task.candidateRecords. Resolve references between cards when needed. The inline lambda itself must return one complete AnswerCandidate record with candidate_id, answer, source_case_id and source_card_ids unchanged. Return that exact selected record.'}),
    ...candidateFiles,
  };
  const groups=[...new Set(selectedRows.flatMap(row=>row.source_groups))];
  const upstreamHashes=[...new Set(selectedRows.flatMap(row=>row.source_hashes))];
  const datasetRecords=[...new Set(selectedRows.flatMap(row=>row.dataset_records))];
  const rawRecordIds=[...new Set(selectedRows.flatMap(row=>row.raw_records.map(record=>record.raw_record_id)))];
  const record=curriculumCase({family:'decision_source_cards',shape:`s${seed}-batch${index}`,variant:'hotpot-reviewed-v1',
    splitGroup:`decision_source_cards:${groups.join('+')}`,slice:'nested_scoped',domain:'other',mode:'single_call',inline:'required',named:'required',
    evidence:{world:batch.map(task=>task.question),retrieved:batch.flatMap(task=>task.sourceCards.map(card=>card.title)),background:[]},
    minimumSequence:['Call the named select_answer child once for each task in batch.',
      'Inside that child, compose the inline question using task.question at runtime and make sourceCards and candidateRecords available to the scoped lambda.',
      'Collect answers with Promise.all and merge by original ordinal, returning each exact structured candidate record.'],
    plausibleActions:['select the candidate answer directly supported by the source cards','select a related but unsupported answer or claim that evidence is insufficient'],
    reference:{root:[evalCall(rootCode),returnCall(expected)],children},
    root:{name:'answer_source_batch',args:{batch:'SourceTask[]'},returns:'AnswerCandidate[]',
      instructions:'For every task in batch, call the named select_answer helper exactly once. It must formulate an inline natural-language decision from that task’s runtime question, titled source cards, and candidate answer records. The source cards may jointly support a comparison or a multi-hop answer. Collect all calls concurrently with Promise.all and return the resulting exact structured candidate records in batch order.'},
    files,inputs:{batch},expected,split});
  record.license=FIXTURE.license;
  record.attribution=FIXTURE.attribution;
  record.dataset=FIXTURE.dataset;
  record.dataset_records=datasetRecords;
  record.source_groups=groups;
  record.source_ids=rawRecordIds;
  record.source_revisions=[...new Set([...record.source_revisions,`hotpotqa/${FIXTURE.upstream_revision}`,`reviewed-proposals/${FIXTURE.reviewed_proposals_sha256}`])];
  record.gold_sources=['hotpotqa-answer','hotpotqa-supporting-facts','independent-source-quality-review'];
  record.generation.source_cards={version:1,upstream_revision:FIXTURE.upstream_revision,upstream_split:'train',license:FIXTURE.license,
    attribution:FIXTURE.attribution,proposals_sha256:FIXTURE.reviewed_proposals_sha256,fixture_sha256:sha(FIXTURE_BYTES),
    acquisition_manifest_gap:FIXTURE.acquisition_manifest_gap,source_hashes:upstreamHashes,raw_record_ids:rawRecordIds,
    source_groups:groups,source_row_hashes:selectedRows.flatMap(row=>row.raw_records.map(source=>source.row_sha256_converted_jsonl)),
    variants_are_independent_facts:false,source_generation_candidate_only:true,training_admission:false,
    protected_exact_id_or_question_overlaps:0,known_holds:FIXTURE.known_holds,held_proposals:FIXTURE.held_proposals??[]};
  record.generation.decision_distillation={version:1,domain:'source_card_answer_selection',
    instruction_source:'runtime task question, exact titled source cards and candidate answer records',
    oracle:'reviewed HotpotQA answer/supporting-facts labels; exact supplied candidate copied by code',
    requests:items.map(item=>{const criteria=Object.fromEntries(item.task.candidateRecords.map(candidate=>[candidate.candidate_id,candidate.answer]));
      return {id:`${record.id}:${item.task.proposal_id}`,state:{question:item.task.question,sourceCards:item.task.sourceCards,candidateRecords:item.task.candidateRecords},
        questions:{decision:{type:'choice',instructions:inlineQuestion,criteria}},expected:item.selected.candidate_id};}),
    capture_names:['question','sourceCards','candidateRecords']};
  record.generation.source_provenance=selectedRows.map(row=>({proposal_id:row.proposal_id,source_case_id:row.source_case_id,
    source_hashes:row.source_hashes,source_groups:row.source_groups,raw_records:row.raw_records,
    support_review:row.support_review,reviewed_snapshot_sha256:row.reviewed_snapshot_sha256}));
  record.curriculum.source_review='source-reviewed exact HotpotQA train records; no generated model output or admission';
  record.curriculum.source_variant_pair=pair;
  return [record];
}
