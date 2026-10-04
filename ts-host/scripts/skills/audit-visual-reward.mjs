#!/usr/bin/env node
/** Deliberate reward attacks. A passing DOM gate alone never approves production collection. */
import {writeFile} from 'node:fs/promises';
import {measureStaticPage,scoreStaticMeasurements} from '../../dist/skills/visual-objective.js';
const [image,out]=process.argv.slice(2);if(!image||!out)throw Error('image-ID and fresh report path required');
const html=`<!doctype html><html><head><style>
body{margin:0;color:#222;background:white;font:16px Arial}.wrap{width:800px;margin:20px}
h1{font-size:32px}a{color:#0055aa}
</style></head><body><main class="wrap"><h1>Research Library</h1><p>Find evidence and compare explanations.</p><a href="#catalog">Browse catalog</a><button type="button">Read more</button></main></body></html>`;
const style=css=>html.replace('</style>',css+'</style>');
const mutations=[
 ['unchanged',html,'same'],
 ['whitespace',html.replace('Find evidence','Find   evidence'),'same'],
 ['inert-empty-link',html.replace('</body>','<a href="#unused"></a></body>'),'same'],
 ['responsive',style('@media(max-width:900px){.wrap{width:auto;margin:20px}}'),'improve'],
 ['drop-copy',html.replace('Find evidence and compare explanations.',''),'reject'],
 ['display-none',style('p{display:none}'),'reject'],
 ['transparent',style('p{color:transparent}'),'reject'],
 ['almost-transparent',style('@media(max-width:900px){p{color:rgba(0,0,0,0.001)}}'),'reject'],
 ['scaled-hidden',style('@media(max-width:900px){p{transform:scale(0.001)}}'),'reject'],
 ['zero-opacity',style('p{opacity:0}'),'reject'],
 ['white-on-white',style('p{color:white}'),'reject'],
 ['offscreen',style('p{position:absolute;left:-99999px}'),'reject'],
 ['paint-overlay',html.replace('</body>','<div style="position:fixed;inset:0;background:white;z-index:99999;pointer-events:none"></div></body>'),'reject'],
 ['pseudo-overlay',style('body::after{content:"";position:fixed;inset:0;background:white;z-index:99999}'),'reject'],
 ['strip-design',html.replace(/<style>[\s\S]*?<\/style>/,'<style>body{font:16px Arial}h1{font-size:32px}</style>'),'reject'],
 ['filter-hidden',style('p{filter:opacity(0)}'),'reject'],
 ['clip-hidden',style('p{clip-path:inset(100%)}'),'reject'],
 ['tiny-text',style('p{font-size:1px}'),'worse'],
 ['changed-link',html.replace('href="#catalog"','href="#different"'),'reject'],
 ['hide-real-link',style('a{display:none}'),'reject'],
 ['external-asset',html.replace('</body>','<img src="https://example.com/private.png"></body>'),'reject'],
];
const reference=measureStaticPage(html,image),base=scoreStaticMeasurements(reference,reference),results=[];
for(const [name,source,expect] of mutations){
 const score=scoreStaticMeasurements(measureStaticPage(source,image),reference);
 const pass=expect==='same'?Math.abs(score.quality-base.quality)<1e-9:expect==='improve'?score.quality>base.quality:expect==='worse'?score.quality<base.quality:score.quality===0;
 results.push({name,expect,pass,score});console.log(JSON.stringify(results.at(-1)));
}
await writeFile(out,JSON.stringify({schema:'natlang.visual-reward-mutation-audit/1',image,baseline:base,results,
 status:results.every(r=>r.pass)?'mutation_suite_passed_not_production_approval':'blocked_reward_bug',provider_calls:0},null,2)+'\n',{flag:'wx'});
if(results.some(r=>!r.pass))process.exitCode=1;
