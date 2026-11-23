import test from 'node:test';import assert from 'node:assert/strict';
import {rewriteTypeModuleSpecifiers,packageExportTarget} from '../dist/index.js';
test('type links rewrite real TS type module specifiers while preserving runtime imports and prose',()=>{
 const source=`// import type {T} from "pkg/types";\nconst prose='import type {T} from "pkg/types"';\nimport value from "pkg/types";\nimport type {T} from "pkg/types";\nimport {type U} from "pkg/types";\nexport type {T} from "pkg/types";\nexport {type U} from "pkg/types";\ntype X=import("pkg/types").T;\nimport type Alias=require("pkg/types");`;
 const actual=rewriteTypeModuleSpecifiers(source,specifier=>specifier==='pkg/types'?'../reviewer/types.js':undefined);
 assert.equal(actual.split('"../reviewer/types.js"').length-1,6);assert.ok(actual.includes('import value from "pkg/types"'));assert.ok(actual.includes("const prose='import type {T} from \"pkg/types\"'"));assert.ok(actual.startsWith('// import type {T} from "pkg/types";'));
});
test('workspace type export resolution shares exact/wildcard public export semantics and explicit conditions',()=>{
 const exports={'.':{types:'./types.ts',import:'./index.ts'},'./types':'./types.ts','./models/*':'./models/*.ts'};
 assert.equal(packageExportTarget(exports,'.',['types','import','default']),'./types.ts');assert.equal(packageExportTarget(exports,'.'),'./index.ts');assert.equal(packageExportTarget(exports,'./models/small'),'./models/small.ts');assert.equal(packageExportTarget(exports,'./private'),undefined);
});

test('type linking preserves JSX literal text and rewrites only its actual type import',()=>{
 const source=`import type {T} from "pkg/types";const element=<div title='import type {T} from "pkg/types"'>text</div>;`;
 const result=rewriteTypeModuleSpecifiers(source,value=>value==='pkg/types'?'./types.js':undefined,'component.tsx');
 assert.ok(result.startsWith('import type {T} from "./types.js";'));assert.ok(result.includes('title=\'import type {T} from "pkg/types"\''));
});
