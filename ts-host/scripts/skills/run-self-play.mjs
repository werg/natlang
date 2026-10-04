#!/usr/bin/env node
/** Exercise real model players; host-private match records remain unpublished until admission. */
import {mkdir,writeFile,appendFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {createHash} from 'node:crypto';
import {wordGames,wordScenarioCases} from '../../dist/self-play/word-games.js';
import {semanticGames,semanticScenarioCases} from '../../dist/self-play/semantic-games.js';
import {createChesstGame,createChesstTacticFixture} from '../../dist/self-play/chesst.js';
import {playMatch,replayMatch} from '../../dist/self-play/arena.js';
import {natlangGamePolicy,GAME_POLICY_SOURCE} from '../../dist/self-play/policy.js';
import {openAICompatibleModelTurn} from '../../dist/model/openai-compatible.js';
import {recordingModelDriver} from './record-model-turn.mjs';
const options={endpoint:'http://127.0.0.1:8082',model:'nvidia/Qwen3.6-35B-A3B-NVFP4',seed:101,workers:2,'cases-per-game':1,'max-decisions':64};
for(let i=2;i<process.argv.length;i+=2){const key=process.argv[i].replace(/^--/,''),value=process.argv[i+1];
  if(!['out','endpoint','model','seed','workers','cases-per-game','max-decisions','chesst-source','games'].includes(key)||value===undefined)throw Error('unknown/missing option '+key);
  options[key]=['seed','workers','cases-per-game','max-decisions'].includes(key)?Number(value):value;}
if(!options.out||!Number.isSafeInteger(options.seed)||['workers','cases-per-game','max-decisions'].some(key=>!Number.isSafeInteger(options[key])||options[key]<1)||options.workers>16)throw Error('out and finite valid allocations required');
const out=resolve(options.out);await mkdir(out,{recursive:false});
const games=new Map([...semanticGames,...wordGames].map(game=>[game.id,game]));
const counts=new Map();const selected=[...semanticScenarioCases(),...wordScenarioCases()].filter(row=>{const n=counts.get(row.family)||0;counts.set(row.family,n+1);return n<options['cases-per-game'];});
if(options['chesst-source']){
  const game=await createChesstGame(options['chesst-source']);games.set(game.id,game);
  // Short forced tactics finish under actual victory rules; no cap-derived rewards.
  const base=createChesstTacticFixture();base.board=Array.from({length:8},()=>Array(8).fill(null));
  base.board[4][4]={type:'king',color:'gold',hasMoved:true};base.board[3][4]={type:'king',color:'blue',hasMoved:true};
  selected.push({group:'chesst/local-royal-ransom',family:'chesst',scenario:base});
}
if(options.games){const requested=options.games.split(',');if(requested.some(id=>!games.has(id)))throw Error('unknown requested game');
  for(let index=selected.length-1;index>=0;index--)if(!requested.includes(selected[index].family))selected.splice(index,1);}
if(!selected.length)throw Error('no selected games');
const policy={files:{'play.nl':GAME_POLICY_SOURCE},entry:'play.nl',model:`${options.endpoint}:${options.model}`};
const controller=new AbortController();for(const signal of ['SIGINT','SIGTERM'])process.once(signal,()=>controller.abort(Error(signal)));
const manifest={schema:'natlang.self-play-exercise/1',options,policy,games:Object.fromEntries([...games].map(([id,game])=>[id,{revision:game.revision,source:game.source??null}])),groups:selected.map(row=>row.group),publication:'Unpublished host-private R&D evidence; not an SFT admission'};
await writeFile(join(out,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});
const results=[];let next=0;
await Promise.all(Array.from({length:Math.min(options.workers,selected.length)},async()=>{
  while(next<selected.length&&!controller.signal.aborted){const index=next++,row=selected[index],game=games.get(row.family),directory=join(out,String(index).padStart(3,'0'));await mkdir(directory);
    const driver=recordingModelDriver({createDriver:onExchange=>openAICompatibleModelTurn({endpoint:options.endpoint,model:options.model,request:{temperature:.2},apiKey:process.env.NATLANG_IMPROVEMENT_API_KEY,onExchange}),
      record:exchange=>appendFile(join(directory,'effective-exchanges.jsonl'),JSON.stringify(exchange)+'\n'),recordWire:exchange=>appendFile(join(directory,'wire-exchanges.jsonl'),JSON.stringify(exchange)+'\n')});
    const policies=Object.fromEntries(game.seats.map(seat=>[seat,natlangGamePolicy({snapshot:policy,driver,seed:options.seed+index,
      onDecision:audit=>appendFile(join(directory,'private-decisions.jsonl'),JSON.stringify({seat,...audit})+'\n')})]));
    const started=Date.now();const match=await playMatch({game,scenario:row.scenario,seed:options.seed+index,policies,maxDecisions:options['max-decisions'],signal:controller.signal});
    await writeFile(join(directory,'match.json'),JSON.stringify(match,null,2)+'\n',{flag:'wx'});
    let replayed=false,replayError=null;if(match.disposition==='completed'){try{await replayMatch(game,match);replayed=true;}catch(error){replayError=String(error);}}
    const result={group:row.group,game:game.id,disposition:replayError?'replay-rejected':match.disposition,replayError,scores:match.outcome?.scores??null,error:match.error??null,decisions:match.frames.length,replayed,elapsedMs:Date.now()-started,matchSha256:createHash('sha256').update(JSON.stringify(match)).digest('hex')};
    results.push(result);await appendFile(join(out,'progress.jsonl'),JSON.stringify(result)+'\n');console.log(JSON.stringify(result));
  }
}));
await writeFile(join(out,'summary.json'),JSON.stringify({results,interrupted:controller.signal.aborted,publication:manifest.publication},null,2)+'\n',{flag:'wx'});
