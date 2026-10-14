#!/usr/bin/env node
// Stream JSONL identities using the collector's canonical digest implementation.
import {jsonlLines} from './jsonl-stream.mjs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const args=process.argv.slice(2),options={};
for(let i=0;i<args.length;i+=2){
  if(!['--input','--runtime','--field'].includes(args[i])||!args[i+1])throw new Error('Usage: record-digests.mjs --input PATH [--runtime HOST_ROOT] [--field task.program_ir]');
  options[args[i].slice(2)]=args[i+1];
}
if(!options.input)throw new Error('--input JSONL path is required');
const host=path.resolve(options.runtime??path.join(import.meta.dirname,'..'));
const {recordDigest}=await import(pathToFileURL(path.join(host,'dist/teacher/collector.js')));
let physicalLine=0;
for await(const line of jsonlLines(options.input)){
  physicalLine++;
  if(!line.trim())continue;
  let record=JSON.parse(line);
  for(const key of options.field?.split('.')??[])record=record?.[key];
  if(!record||typeof record!=='object'||Array.isArray(record))throw new Error(`Missing record at physical line ${physicalLine}`);
  const output=JSON.stringify({physical_line:physicalLine,id:record.id??null,record_digest:recordDigest(record)})+'\n';
  if(!process.stdout.write(output))await new Promise(resolve=>process.stdout.once('drain',resolve));
}
