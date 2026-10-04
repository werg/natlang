/** Host-owned static-page measurements. No reference HTML is passed to the browser. */
import {spawnSync} from 'node:child_process';
import {randomBytes,createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';

export type VisualBox = {x:number;y:number;width:number;height:number};
export type VisualText = {text:string;visible:boolean;opacity:number;fontSize:number;color:string;background:string;clipped:boolean;boxes:VisualBox[]};
export type VisualPage = {width:number;scrollWidth:number;scrollHeight:number;title:string;texts:VisualText[];
  anchors:{tag:string;text:string;type:string|null;href:string|null;name:string|null;visible:boolean;box:VisualBox}[];
  blockedResources:string[];unsupportedPaint:string[]};
export type VisualMeasurements = {schema:'natlang.static-browser-measurements/1';pages:VisualPage[]};
export type VisualObjective = {schema:'natlang.static-page-objective/1';image:string;reference:VisualMeasurements};

/** These limits allocate generation resources; they are not language execution semantics. */
export function measureStaticPage(html:string,image:string):VisualMeasurements {
  if(!/^sha256:[a-f0-9]{64}$/.test(image)) throw Error('visual renderer requires an immutable Docker image ID');
  if(Buffer.byteLength(html)>192*1024) throw Error('visual artifact exceeds allocation (unscored)');
  const seccomp=fileURLToPath(new URL('../../scripts/skills/visual-browser-seccomp.json',import.meta.url));
  if(createHash('sha256').update(readFileSync(seccomp)).digest('hex')!=='cc3e61cabda6bbc1e53e54d27ba4d55a9d3be829b6dd1a596f4a7b31b1cc7849')
    throw Error('visual sandbox seccomp pin mismatch (unscored)');
  const token=randomBytes(24).toString('hex'),name='natlang-visual-'+token;
  try {
    const result=spawnSync('docker',['run','--rm','--name',name,'-i','--init','--network','none',
      '--read-only','--memory','1g','--memory-swap','1g','--cpus','1','--pids-limit','128',
      '--shm-size','128m','--tmpfs','/tmp:rw,nosuid,size=128m','--cap-drop','ALL','--cap-add','SYS_CHROOT',
      '--security-opt','no-new-privileges','--security-opt','seccomp='+seccomp,'--user','pwuser',image],
      {input:JSON.stringify({html,token}),encoding:'utf8',timeout:45000,maxBuffer:2*1024*1024});
    if(result.error||result.status!==0) throw Error('visual renderer infrastructure failure (unscored): '+
      String(result.error?.message||result.stderr||result.signal).slice(0,700));
    const reply=JSON.parse(result.stdout);
    if(reply.token!==token||reply.schema!=='natlang.static-browser-measurements/1'||
      !Array.isArray(reply.pages)||reply.pages.map((p:VisualPage)=>p.width).join(',')!=='360,768,1280')
      throw Error('invalid visual measurement protocol (unscored)');
    return {schema:reply.schema,pages:reply.pages};
  } finally {spawnSync('docker',['rm','-f',name],{timeout:5000,stdio:'ignore'});}
}

const words=(page:VisualPage)=>page.texts.map(t=>t.text).join(' ').replace(/\s+/gu,' ').trim();
const semantics=(page:VisualPage)=>page.anchors.map(a=>JSON.stringify([a.tag,a.text,a.type,a.href,a.name])).sort().join('\n');
/** Conservative pilot objective. Desktop geometry guard is a contract, not an aesthetic judgment. */
export function scoreStaticMeasurements(candidate:VisualMeasurements,reference:VisualMeasurements) {
  if(reference.schema!=='natlang.static-browser-measurements/1'||reference.pages.length!==3)
    throw Error('invalid visual reference (unscored)');
  const wide=reference.pages[2]!;
  if(!wide.texts.length||wide.blockedResources.length||wide.unsupportedPaint?.length||wide.texts.some(t=>!t.visible))
    throw Error('unsupported visual source reference (unscored)');
  const gates={visible_content:true,semantic_affordances:true,self_contained:true,desktop_geometry:true,desktop_text_style:true};
  let worstViewportPenalty=0;
  for(const page of candidate.pages){
    gates.visible_content &&= words(page)===words(wide)&&page.texts.every(t=>t.visible&&t.boxes.length>0);
    gates.semantic_affordances &&= semantics(page)===semantics(wide)&&page.anchors.every(a=>a.visible);
    gates.self_contained &&= page.blockedResources.length===0;
    gates.visible_content &&= Array.isArray(page.unsupportedPaint)&&page.unsupportedPaint.length===0;
    // Worst-viewport scoring avoids diluting one broken layout with many DOM nodes.
    const overflow=Math.min(1,Math.max(0,page.scrollWidth-page.width)/page.width);
    let clipping=0,legibility=0;
    for(const text of page.texts){
      const height=Math.min(...text.boxes.map(b=>b.height));
      legibility+=Math.min(1,Math.max(0,12-Math.min(text.fontSize,height))/12);
      clipping+=Number(text.clipped);
    }
    worstViewportPenalty=Math.max(worstViewportPenalty,0.6*overflow+
      0.25*clipping/Math.max(1,page.texts.length)+0.15*legibility/Math.max(1,page.texts.length));
  }
  const desktop=candidate.pages[2]!;
  const tokenStyles=(page:VisualPage)=>page.texts.flatMap(t=>t.text.split(/\s+/u).map(()=>t));
  const originalTokens=tokenStyles(wide),candidateTokens=tokenStyles(desktop);
  if(originalTokens.length!==candidateTokens.length)gates.desktop_text_style=false;
  else for(let i=0;i<originalTokens.length;i++){
    const a=originalTokens[i]!,b=candidateTokens[i]!;
    if(a.color!==b.color||a.background!==b.background||b.fontSize<a.fontSize*0.8||b.fontSize>a.fontSize*1.2)
      gates.desktop_text_style=false;
    const aa=a.boxes[0],bb=b.boxes[0];
    if(!aa||!bb||Math.abs(aa.x-bb.x)>128||Math.abs(aa.y-bb.y)>Math.max(90,wide.scrollHeight*0.1))gates.desktop_geometry=false;
  }
  // Preserve identities and rough desktop placement of semantic anchors.
  if(desktop.anchors.length!==wide.anchors.length) gates.desktop_geometry=false;
  else for(let i=0;i<wide.anchors.length;i++){
    const a=wide.anchors[i]!,b=desktop.anchors[i]!;
    if(Math.abs(a.box.x-b.box.x)>128||Math.abs(a.box.y-b.box.y)>Math.max(90,wide.scrollHeight*0.1)||
      Math.abs(a.box.width-b.box.width)>Math.max(128,a.box.width*0.3)) gates.desktop_geometry=false;
  }
  return {quality:Object.values(gates).every(Boolean)?1-worstViewportPenalty:0,gates};
}
export function scoreStaticPage(value:unknown,expected:unknown) {
  if(typeof value!=='string')return {quality:0,gates:{html_string:false}};
  const task=expected as VisualObjective;
  if(task?.schema!=='natlang.static-page-objective/1') throw Error('invalid visual task (unscored)');
  const fence=/^\s*```(?:html)?\s*([\s\S]*?)```\s*$/i.exec(value);
  return scoreStaticMeasurements(measureStaticPage(fence?fence[1]!:value,task.image),task.reference);
}
