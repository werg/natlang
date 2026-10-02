#!/usr/bin/env node
/** Upgrade reviewed task contracts before assigning IR from an older snapshot. */
import {createReadStream} from 'node:fs';
import {createInterface} from 'node:readline';
import {open,rename,unlink,writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
import {applyEvidenceScaleContract,applyTatqaNumericContract,applyReviewedTatqaUnitContract} from '../ts-host/scripts/inline-curriculum/directory-sources.mjs';
import {applyReviewedMusiqueOracleAlias} from '../ts-host/scripts/inline-curriculum/directory-sources.mjs';
import {applyReviewedMusiqueOutputEquivalence,reviewedOutputEquivalenceEntry} from '../ts-host/scripts/inline-curriculum/musique-reviewed-output-equivalences.mjs';
import {applyRecentMusiqueOutputEquivalences} from '../ts-host/scripts/inline-curriculum/musique-recent-output-equivalences.mjs';
import {applyMinimumAgeReviewedContract,validateMinimumAgeReviewedVariant} from '../ts-host/scripts/inline-curriculum/musique-minimum-age-reviewed.mjs';
import {applyReviewedOklahomaAnnualEventContract} from '../ts-host/scripts/inline-curriculum/musique-oklahoma-annual-event-reviewed.mjs';
import {applyTatqaProportionDisplay,tatqaProportionDisplayPins} from '../ts-host/scripts/inline-curriculum/tatqa-proportion-display-reviewed.mjs';
import {sourceConversionDigest} from '../ts-host/dist/teacher/source-conversion.js';

const [inputArg,outputArg]=process.argv.slice(2);
if(!inputArg||!outputArg)throw Error('usage: prepare_generation_ir INPUT.ir.jsonl NEW_OUTPUT.ir.jsonl');
const input=resolve(inputArg),output=resolve(outputArg);
if(input===output)throw Error('Preserve the original IR; choose a new output path');
const receiptPath=output+'.receipt.json';
// Exclusive creation prevents replacing a frozen assignment or an earlier review.
const reservation=await open(output,'wx');await reservation.close();
let receiptFile;
try { receiptFile=await open(receiptPath,'wx'); }
catch(error){ await unlink(output).catch(()=>{});throw error; }
const temporary=output+'.preparing';let stream,temporaryOwned=false;
const sourceHash=createHash('sha256'),outputHash=createHash('sha256');
let cases=0,changed=0;
const changedCases=[];
const transforms={tatqa_contracts:0,musique_aliases:0,musique_oklahoma:0,musique_output_equivalence_v1:0,
  musique_output_equivalence_v2:0,musique_minimum_age:0};
// The legacy assignment has complete group assembly. These five source-pinned
// alias contracts are allowed to transform only these exact modern base rows;
// the adapter modules independently pin their source snapshots and evidence.
const ALIAS_BASE_DIGESTS=Object.freeze({
  '2hop__128979_90736':'e873bf99749072e9b45c5b3ee37972f8f472a30035a86f6feee40a288d73f981',
  '4hop1__205937_144938_83779_69861':'3603ae25b974851f9ad3396573012016c47d19dbba4360f02c6769f2938814b9',
  '3hop1__478606_751065_78953':'5f689c01ec3f1cfdafff5548d5ad83e0a40a3f4aa23e6df4b43277df15b711ba',
  '2hop__136323_160978':'da087b96b872bc611914938e8359f91e6bcaae77bd0ec1418985bdacdd25d689',
  '2hop__130422_69489':'1dc820e0e9f3b8cd2e1a07175ce57ea0f216c40c93c6ad07386ef8c71d178f4d',
});
const ALIAS_BASE_IDS=Object.freeze({
  '2hop__128979_90736':'inline-curriculum:source_musique:0ede675c65f8cc5c9964:v1',
  '4hop1__205937_144938_83779_69861':'inline-curriculum:source_musique:923205df863c88adc14f:v1',
  '3hop1__478606_751065_78953':'inline-curriculum:source_musique:9a951f540d5d4f0efb69:v1',
  '2hop__136323_160978':'inline-curriculum:source_musique:edc16cf1494f27a7741d:v1',
  '2hop__130422_69489':'inline-curriculum:source_musique:e403cfa783ea6d1932b0:v1',
});
const ALIAS_VARIANT_DIGESTS=Object.freeze({
  '2hop__128979_90736':'2928032d471904a11badfd82dfb421879bcb67a48cf34e6ae5c73109989475c9',
  '4hop1__205937_144938_83779_69861':'49e860b211618aeeee3c44d58bb8d93f65abef956cd56e55767afd8c9f2ac24b',
  '3hop1__478606_751065_78953':'6081840f515ad30d419aaafebafc3e34a8b53d50e5cf774bc423a9b499fcad9f',
  '2hop__136323_160978':'6ddcebc46a23534e6525f3850b86e11629a5cf4dec0316eeee47cf0844e950d5',
  '2hop__130422_69489':'b3cff9c4c8c459229ffdd25d9198674e3db57cf05a413fca5c683c2ce25893d6',
});
const ALIAS_SUFFIXES=Object.freeze({
  '2hop__128979_90736':':reviewed-alias-v1',
  '4hop1__205937_144938_83779_69861':':reviewed-alias-v1',
  '3hop1__478606_751065_78953':':reviewed-alias-v1',
  '2hop__136323_160978':':reviewed-oracle-alias-removal-v2',
  '2hop__130422_69489':':reviewed-oracle-alias-removal-v2',
});
const reviewedMutationProjection=record=>{
  const value=structuredClone(record);delete value.id;delete value.semantics?.oracle;
  for(const key of ['oracle_review','oklahoma_annual_event_review','reviewed_answer_equivalence','minimum_age_equivalence_review']){
    if(value.generation)delete value.generation[key];if(value.external_source)delete value.external_source[key];
  }
  return value;
};
// This already-reviewed exact-operation derivative has additional ID suffixes
// and deliberately replaces the rounding sentence. Preserve its full pinned IR.
const preservedDerivedContracts=new Set(['07a23b35f0d562977d1b1626530ebdf5ec85f16a9f1b6ed7af0f023e9731016c']);
const invariants=ir=>sourceConversionDigest({expected:ir.semantics?.expected,
  expected_files:ir.semantics?.expected_files,inputs:ir.semantics?.inputs,
  folder_files:ir.semantics?.folder_files,reference:ir.curriculum?.reference,
  source_ids:ir.source_ids,source_groups:ir.source_groups,source_revisions:ir.source_revisions,
  external_files:ir.external_source?.files,external_revision:ir.external_source?.revision,
  external_snapshot:ir.external_source?.snapshot_sha256,license:ir.license});
try{
  stream=await open(temporary,'wx');
  temporaryOwned=true;
  for await(const line of createInterface({input:createReadStream(input),crlfDelay:Infinity})){
    if(!line.trim())continue;
    sourceHash.update(line+'\n');
    const ir=JSON.parse(line),beforeRecord=structuredClone(ir),before=sourceConversionDigest(ir),fixed=invariants(ir);
    if(!preservedDerivedContracts.has(before))
      {
        const prior=sourceConversionDigest(ir);
        applyReviewedTatqaUnitContract(applyTatqaNumericContract(applyEvidenceScaleContract(ir)));
        if(sourceConversionDigest(ir)!==prior)transforms.tatqa_contracts++;
      }
    if(tatqaProportionDisplayPins().some(pin=>pin.sourceId===ir.source_ids?.[0]))
      Object.assign(ir,applyTatqaProportionDisplay(ir));
    if(ir.source==='musique'){
      const sourceId=ir.external_source?.source_id??ir.source_ids?.[0];
      const aliasBase=ALIAS_BASE_DIGESTS[sourceId],aliasVariant=ALIAS_VARIANT_DIGESTS[sourceId];
      if(aliasBase){
        const ageReviewed=sourceId==='2hop__130422_69489' &&
          (ir.generation?.minimum_age_equivalence_review!==undefined || ir.external_source?.minimum_age_equivalence_review!==undefined);
        if(ageReviewed){
          // The approved age contract is composed after alias removal. Its own
          // validator pins the entire final canonical or historical titled IR.
          if(!validateMinimumAgeReviewedVariant(ir))
            throw Error(`reviewed_musique_alias_minimum_age_composition_mismatch:${sourceId}`);
        }else{
          const baseId=ALIAS_BASE_IDS[sourceId],suffix=ALIAS_SUFFIXES[sourceId];
          const isBase=ir.id===baseId,isVariant=ir.id===`${baseId}${suffix}`;
          if(!isBase&&!isVariant)
            throw Error(`reviewed_musique_alias_unrecognized_id:${sourceId}:${ir.id}`);
          const digestBefore=sourceConversionDigest(ir);
          if(isBase&&digestBefore!==aliasBase)
            throw Error(`reviewed_musique_alias_full_base_digest_mismatch:${sourceId}`);
          if(isVariant&&digestBefore!==aliasVariant)
            throw Error(`reviewed_musique_alias_full_variant_digest_mismatch:${sourceId}`);
          const prior=structuredClone(ir),priorDigest=sourceConversionDigest(ir);
          applyReviewedMusiqueOracleAlias(ir);
          if(sourceConversionDigest(ir)!==priorDigest){
            if(sourceConversionDigest(reviewedMutationProjection(prior))!==sourceConversionDigest(reviewedMutationProjection(ir)))
              throw Error(`reviewed_musique_alias_unexpected_mutation:${sourceId}`);
            if(sourceConversionDigest(ir)!==aliasVariant)
              throw Error(`reviewed_musique_alias_unpinned_output_digest:${sourceId}`);
            transforms.musique_aliases++;
          }
        }
      } else {
        // The central adapter remains authoritative for any other registered row.
        const priorDigest=sourceConversionDigest(ir);
        applyReviewedMusiqueOracleAlias(ir);
        if(sourceConversionDigest(ir)!==priorDigest)
          throw Error(`reviewed_musique_alias_missing_local_full_digest_pin:${sourceId}`);
      }
      const applyGuarded=(label,fn,counter)=>{
        const prior=structuredClone(ir),priorDigest=sourceConversionDigest(ir);
        const returned=fn(ir);
        if(returned!==ir)Object.assign(ir,returned);
        if(sourceConversionDigest(ir)!==priorDigest){
          if(sourceConversionDigest(reviewedMutationProjection(prior))!==sourceConversionDigest(reviewedMutationProjection(ir)))
            throw Error(`${label}_unexpected_mutation:${sourceId}`);
          transforms[counter]++;
        }
      };
      applyGuarded('oklahoma',applyReviewedOklahomaAnnualEventContract,'musique_oklahoma');
      const eq=reviewedOutputEquivalenceEntry(sourceId);
      if(eq)applyGuarded('output_equivalence_v1',row=>applyReviewedMusiqueOutputEquivalence(row,eq.accepted),'musique_output_equivalence_v1');
      applyGuarded('output_equivalence_v2',applyRecentMusiqueOutputEquivalences,'musique_output_equivalence_v2');
      applyGuarded('minimum_age',applyMinimumAgeReviewedContract,'musique_minimum_age');
    }
    if(invariants(ir)!==fixed)throw Error('Contract migration changed gold, input, reference or source lineage');
    const next=sourceConversionDigest(ir);if(next!==before){changed++;changedCases.push({source_id:ir.source_ids?.[0],source:ir.source,
      base_id:beforeRecord.id,variant_id:ir.id,base_ir_sha256:before,variant_ir_sha256:next});}
    const bytes=JSON.stringify(ir)+'\n';outputHash.update(bytes);await stream.writeFile(bytes);cases++;
  }
  await stream.sync();await stream.close();stream=undefined;
  await rename(temporary,output);
  const receipt={version:'natlang.generation_ir_preparation/2',created_at:new Date().toISOString(),input,output,cases,changed,
    normalized_input_lines_sha256:sourceHash.digest('hex'),output_sha256:outputHash.digest('hex'),
    transforms,changed_cases:changedCases,
    changes:'Existing reviewed TATQA contracts and exact source-pinned MuSiQue alias/Oklahoma/output-equivalence/minimum-age contracts; versioned IDs',
    gold_input_reference_lineage_preserved:true,ready_for_generation:false,
    next_step:'Review native gold replay, current admission/materialization, selection lineage and frozen runtime pins before launch.'};
  await receiptFile.writeFile(JSON.stringify(receipt,null,2)+'\n');await receiptFile.sync();await receiptFile.close();receiptFile=undefined;
  console.log(JSON.stringify(receipt));
}catch(error){
  if(stream)await stream.close();if(receiptFile)await receiptFile.close();if(temporaryOwned)await unlink(temporary).catch(()=>{});await unlink(output).catch(()=>{});await unlink(receiptPath).catch(()=>{});throw error;
}
