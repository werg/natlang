#!/usr/bin/env node
/** Upgrade reviewed task contracts before assigning IR from an older snapshot. */
import {createReadStream} from 'node:fs';
import {createInterface} from 'node:readline';
import {open,rename,unlink} from 'node:fs/promises';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
import {applyEvidenceScaleContract,applyTatqaNumericContract,applyReviewedTatqaUnitContract} from '../ts-host/scripts/inline-curriculum/directory-sources.mjs';
import {sourceConversionDigest} from '../ts-host/dist/teacher/source-conversion.js';

const [inputArg,outputArg]=process.argv.slice(2);
if(!inputArg||!outputArg)throw Error('usage: prepare_generation_ir INPUT.ir.jsonl NEW_OUTPUT.ir.jsonl');
const input=resolve(inputArg),output=resolve(outputArg);
if(input===output)throw Error('Preserve the original IR; choose a new output path');
// Exclusive creation prevents replacing a frozen assignment or an earlier review.
const reservation=await open(output,'wx');await reservation.close();
const temporary=output+'.preparing';let stream,temporaryOwned=false;
const sourceHash=createHash('sha256'),outputHash=createHash('sha256');
let cases=0,changed=0;
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
    const ir=JSON.parse(line),before=sourceConversionDigest(ir),fixed=invariants(ir);
    if(!preservedDerivedContracts.has(before))
      applyReviewedTatqaUnitContract(applyTatqaNumericContract(applyEvidenceScaleContract(ir)));
    if(invariants(ir)!==fixed)throw Error('Contract migration changed gold, input, reference or source lineage');
    const next=sourceConversionDigest(ir);if(next!==before)changed++;
    const bytes=JSON.stringify(ir)+'\n';outputHash.update(bytes);await stream.writeFile(bytes);cases++;
  }
  await stream.sync();await stream.close();stream=undefined;
  await rename(temporary,output);
  console.log(JSON.stringify({version:'natlang.generation_ir_preparation/1',input,output,cases,changed,
    normalized_input_lines_sha256:sourceHash.digest('hex'),output_sha256:outputHash.digest('hex'),
    changes:'Existing reviewed TATQA scale/numeric/source-unit contracts; versioned IDs',
    gold_input_reference_lineage_preserved:true,ready_for_generation:false,
    next_step:'Review native gold replay, current admission/materialization, selection lineage and frozen runtime pins before launch.'}));
}catch(error){
  if(stream)await stream.close();if(temporaryOwned)await unlink(temporary).catch(()=>{});await unlink(output).catch(()=>{});throw error;
}
