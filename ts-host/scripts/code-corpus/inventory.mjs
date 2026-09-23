import { readdir, readFile } from 'node:fs/promises';
import { dirname, join, relative, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import ts from 'typescript';
import { extractFunctions } from './extract.mjs';
import { enumerateExercism } from './exercism.mjs';
import { writeJsonl } from './common.mjs';

const ignoredDirectories = new Set(['node_modules','.git','dist','coverage','__tests__']);
const ignoredCodeFile = (name) => /(?:\.(?:spec|test|d)\.[cm]?[jt]s|_test\.[cm]?ts)$/.test(name)
  || /(?:^|\.)config\.[cm]?[jt]s$/.test(name)
  || /(?:^|\.)conf\.[cm]?[jt]s$/.test(name);

async function inventoryGeneric(root, metadata, {limit = 1000} = {}) {
  if (!Number.isInteger(limit) || limit < 1) throw new Error('positive integer limit required');
  const rows = [];
  async function walk(dir) {
    for (const entry of (await readdir(dir,{withFileTypes:true})).sort((a,b)=>a.name.localeCompare(b.name))) {
      if (rows.length >= limit) return;
      if (entry.isSymbolicLink() || ignoredDirectories.has(entry.name)) continue;
      const path = join(dir,entry.name);
      if (entry.isDirectory()) await walk(path);
      else if (/\.[cm]?[jt]s$/.test(entry.name) && !ignoredCodeFile(entry.name)) {
        const rel = relative(root,path);
        let nearestReadme;
        if (metadata.sourceName === 'deno-std' || metadata.sourceName === 'javascript-algorithms') {
          let parent = dirname(path);
          while (parent === root || parent.startsWith(`${root}/`)) {
            try {
              const readmePath = join(parent, 'README.md');
              const text = await readFile(readmePath, 'utf8');
              nearestReadme = { path: relative(root, readmePath), text };
              break;
            } catch (error) { if (error.code !== 'ENOENT') throw error; }
            if (parent === root) break;
            parent = dirname(parent);
          }
        }
        const source = await readFile(path,'utf8');
        let functions = extractFunctions(source,{...metadata,path:rel});
        if (!functions.length && nearestReadme?.text.trim()) {
          const instruction = `Context from ${nearestReadme.path}:\n\n${nearestReadme.text}`;
          functions = extractFunctions(source,{...metadata,path:rel,instruction}).map((record) => ({
            ...record,
            raw: { readmePath: nearestReadme.path, readme: nearestReadme.text, alignment: 'file-level README context; function-level alignment not verified' },
            verification: { ...record.verification, status: 'inventory', reasons: [...(record.verification.reasons ?? []), 'file-level README context; function alignment not verified'] },
          }));
        }
        rows.push(...functions.slice(0,limit-rows.length));
      }
    }
  }
  await walk(root); return rows;
}

export async function inventory(root, metadata, options = {}) {
  if (metadata.sourceName === '30-seconds-of-code') return inventoryMarkdown(root, metadata, options);
  if (metadata.sourceName === 'd3-array') return inventoryExports(root, metadata, options);
  return inventoryGeneric(root, metadata, options);
}

/** Inventory JS snippets from markdown frontmatter and fences without evaluating them. */
export async function inventoryMarkdown(root, metadata, {limit=1000} = {}) {
  const rows=[];
  async function walk(dir) {
    for (const entry of (await readdir(dir,{withFileTypes:true})).sort((a,b)=>a.name.localeCompare(b.name))) {
      if (rows.length>=limit) return;
      if (entry.isSymbolicLink() || ignoredDirectories.has(entry.name)) continue;
      const path=join(dir,entry.name);
      if (entry.isDirectory()) await walk(path);
      else if (entry.name.endsWith('.md')) {
        const rel=relative(root,path), text=await readFile(path,'utf8');
        const frontmatter=text.match(/^---\s*\n([\s\S]*?)\n---\s*(?:\n|$)/);
        const fields=Object.fromEntries((frontmatter?.[1]??'').split('\n').map(line=>line.match(/^([\w-]+):\s*(.*?)\s*$/)).filter(Boolean).map(([,key,value])=>[key,value.replace(/^(['"])(.*)\1$/,'$2')]));
        const prose=(text.slice(frontmatter?.[0].length??0).split(/```/)[0]??'').trim();
        const instruction=[fields.instruction,fields.title,fields.description,prose].filter((part,index,array)=>part && array.indexOf(part)===index).join('\n\n');
        const fence=/```(javascript|js)\s*\n([\s\S]*?)```/gi;
        let match,index=0;
        while ((match=fence.exec(text)) && rows.length<limit) {
          const code=match[2].trim(); if (!code) continue;
          const records=extractFunctions(code,{...metadata,path:rel,instruction:instruction||`Implement the JavaScript example documented in ${rel}.`});
          if (records.length) rows.push(...records.slice(0,limit-rows.length).map((record)=>({...record,id:`${record.id}:fence-${index}`,group_id:`${metadata.sourceName}:${rel}`,raw:{markdownPath:rel,document:text,frontmatter:fields,code}})));
          else rows.push({version:'natlang.code_task/1',id:`${metadata.sourceName}:${metadata.revision}:${rel}:fence-${index}`,group_id:`${metadata.sourceName}:${rel}`,kind:'function',language:'javascript',instruction:instruction||`Implement the JavaScript example documented in ${rel}.`,source:{name:metadata.sourceName,revision:metadata.revision,path:rel,license:metadata.license},function:{name:fields.title??`snippet_${index}`,parameters:[],body:code,source:match[2].trim()},cases:[],verification:{status:'inventory',reasons:['markdown code snippet; signature not inferred']},raw:{markdownPath:rel,document:text,frontmatter:fields,code}});
          index++;
        }
      }
    }
  }
  await walk(root); return rows;
}

/** D3 modules commonly expose default function expressions and exported arrows. */
export async function inventoryExports(root, metadata, {limit=1000} = {}) {
  const rows=[];
  for (const record of await inventoryGeneric(root,metadata,{limit})) rows.push(record);
  async function walk(dir) {
    for (const entry of (await readdir(dir,{withFileTypes:true})).sort((a,b)=>a.name.localeCompare(b.name))) {
      if (rows.length>=limit) return;
      if (entry.isSymbolicLink() || ignoredDirectories.has(entry.name)) continue;
      const path=join(dir,entry.name);
      if (entry.isDirectory()) await walk(path);
      else if (/\.[cm]?js$/.test(entry.name) && !ignoredCodeFile(entry.name)) {
        const source=await readFile(path,'utf8'), file=ts.createSourceFile(path,source,ts.ScriptTarget.Latest,true,ts.ScriptKind.JS), rel=relative(root,path);
        const stem=entry.name.replace(/\.[cm]?js$/,'');
        let testPath, testDescriptions=[];
        for (const candidate of [`${stem}-test.js`,`${stem}.test.js`]) {
          try {
            const candidatePath=join(root,'test',candidate);
            const testSource=await readFile(candidatePath,'utf8');
            const testFile=ts.createSourceFile(candidatePath,testSource,ts.ScriptTarget.Latest,true,ts.ScriptKind.JS);
            const visit=(node)=>{
              if (ts.isCallExpression(node) && ts.isIdentifier(node.expression) && ['test','it'].includes(node.expression.text)
                && node.arguments[0] && (ts.isStringLiteral(node.arguments[0]) || ts.isNoSubstitutionTemplateLiteral(node.arguments[0]))) testDescriptions.push(node.arguments[0].text);
              ts.forEachChild(node,visit);
            };
            visit(testFile);
            if (testDescriptions.length) { testPath=relative(root,candidatePath); break; }
          } catch (error) { if (error.code !== 'ENOENT') throw error; }
        }
        for (const stmt of file.statements) {
          if (rows.length>=limit) break;
          let name,node;
          if (ts.isFunctionDeclaration(stmt) && stmt.modifiers?.some(m=>m.kind===ts.SyntaxKind.ExportKeyword || m.kind===ts.SyntaxKind.DefaultKeyword)) { name=stmt.name?.text??stem; node=stmt; }
          if (ts.isVariableStatement(stmt) && stmt.modifiers?.some(m=>m.kind===ts.SyntaxKind.ExportKeyword)) for (const decl of stmt.declarationList.declarations) {
            if (ts.isIdentifier(decl.name) && decl.initializer && (ts.isArrowFunction(decl.initializer)||ts.isFunctionExpression(decl.initializer))) { name=decl.name.text; node=decl.initializer; }
          }
          if (!node || rows.some(r=>r.source.path===rel&&r.function.name===name)) continue;
          const params=node.parameters.map(p=>({name:p.name.getText(file),...(p.type?{type:p.type.getText(file)}:{})}));
          const body=ts.isBlock(node.body)?node.body.getText(file):`{ return (${node.body.getText(file)}); }`;
          const escapedName=name.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
          const matchingDescriptions=testDescriptions.filter(description=>new RegExp(`(^|[^\\w$])${escapedName}([^\\w$]|$)`).test(description));
          const documented=Boolean(matchingDescriptions.length);
          rows.push({version:'natlang.code_task/1',id:`${metadata.sourceName}:${metadata.revision}:${rel}:${name}`,group_id:`${metadata.sourceName}:${rel}:${name}`,kind:'function',language:'javascript',instruction:documented?`Behavior described by upstream tests for ${name}:\n${matchingDescriptions.map(description=>`- ${description}`).join('\n')}`:`Implement the exported ${name} function from ${rel}.`,...(!documented?{instruction_quality:'name_only'}:{}),source:{name:metadata.sourceName,revision:metadata.revision,path:rel,license:metadata.license},function:{name,parameters:params,body,source:node.getText(file)},cases:[],verification:{status:'inventory',reasons:[...(documented?['upstream test descriptions inventoried; assertions and expected outputs not extracted']:['exported source extracted; no matching behavior description found']), 'test behavior is descriptive metadata only']},...(documented?{raw:{testPath, testDescriptions:matchingDescriptions}}:{})});
        }
      }
    }
  }
  await walk(root); return rows;
}
export async function inventoryExercism(root, metadata, {limit=1000} = {}) {
  const rows=[];
  for (const exercise of await enumerateExercism(root)) {
    if (rows.length>=limit) break;
    if (!exercise.instruction) continue;
    const functions=extractFunctions(await readFile(join(root,exercise.sourcePath),'utf8'),{
      ...metadata,path:exercise.sourcePath,instruction:exercise.instruction,
    });
    for (const record of functions.slice(0,limit-rows.length)) rows.push({...record,
      group_id:`exercism:${exercise.slug}`,raw:{exercise},
      verification:{...record.verification,reasons:[...(record.verification.reasons??[]),'exercise-level description; function alignment not verified']}});
  }
  return rows;
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const [root, output, limit = '1000'] = process.argv.slice(2);
  if (!root || !output) throw new Error('Usage: inventory.mjs PINNED_CHECKOUT OUTPUT [LIMIT]');
  const manifest = JSON.parse(await readFile(join(root,'.code-corpus-source.json'),'utf8'));
  const enumerate = manifest.source.startsWith('exercism-') ? inventoryExercism : manifest.source === '30-seconds-of-code' ? inventoryMarkdown : manifest.source === 'd3-array' ? inventoryExports : inventory;
  const rows = await enumerate(resolve(root),{sourceName:manifest.source,revision:manifest.commit,license:manifest.license},{limit:Number(limit)});
  await writeJsonl(output,rows);
  console.log(JSON.stringify({functions:rows.length,source:manifest.source,revision:manifest.commit}));
}
