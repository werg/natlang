// Replay-safe synthetic tasks. Keep inputs JSON-compatible and implementations
// within the native replay evaluator's supported scalar/array boundary.
export const SYNTHETIC_CODE_GENERATOR = 'natlang.code_curriculum/2';
export const DEFAULT_MAX_FAMILY_REPETITIONS = 50;

const choose = (rng, values) => values[Math.floor(rng() * values.length)];
const ints = (rng, n, lo = -9, hi = 12) => Array.from({ length: n }, () => lo + Math.floor(rng() * (hi - lo + 1)));
const words = ['alfa', 'éclair', '🪁', 'oak', 'blue', '東京', 'A', 'a'];
const stringArray = (rng, n) => Array.from({ length: n }, () => choose(rng, words));
const nested = (rng, n) => Array.from({ length: n }, () => ints(rng, Math.floor(rng() * 4), -4, 8));

// Reference functions are deliberately separate from the emitted implementations.
export const SYNTHETIC_CODE_FAMILIES = [
  { name:'stable_unique', param:'items', type:'string[]', out:'string[]', difficulty:'basic',
    instruction:'Return the distinct strings in items in first-occurrence order. Compare strings exactly.',
    body:'{ const result = []; for (const item of items) if (result.indexOf(item) === -1) result.push(item); return result; }',
    make:r=>stringArray(r,3+Math.floor(r()*7)), ref:a=>a.reduce((out,x)=>out.includes(x)?out:[...out,x],[]) , edge:[] },
  { name:'prefix_sums', param:'numbers', type:'number[]', out:'number[]', difficulty:'basic',
    instruction:'Return an array whose position i is the sum of numbers from index 0 through i. Empty input returns an empty array.',
    body:'{ let total = 0; const result = []; for (const value of numbers) { total += value; result.push(total); } return result; }',
    make:r=>ints(r,2+Math.floor(r()*7)), ref:a=>a.map((_,i)=>a.slice(0,i+1).reduce((s,x)=>s+x,0)), edge:[] },
  { name:'count_words', param:'text', type:'string', out:'number', difficulty:'basic',
    instruction:'Count runs of non-whitespace characters. Empty and whitespace-only text has count 0; use JavaScript whitespace rules.',
    body:'{ const matches = text.match(/\\S+/gu); return matches ? matches.length : 0; }',
    make:r=>choose(r,['','  \t','red blue','one\t two  three','café 東京']), ref:s=>(s.match(/\S+/gu)??[]).length, edge:'\n\t' },
  { name:'group_count', param:'items', type:'string[]', out:'string[][]', difficulty:'intermediate',
    instruction:'Return [string, decimalCount] pairs for each distinct item, ordered by first appearance. Compare strings exactly; encode counts as base-10 strings.',
    body:'{ const result = []; for (const item of items) { const pair = result.find(entry => entry[0] === item); if (pair) pair[1] = String(Number(pair[1]) + 1); else result.push([item, "1"]); } return result; }',
    make:r=>stringArray(r,3+Math.floor(r()*7)), ref:a=>{const m=new Map(); for(const x of a)m.set(x,(m.get(x)??0)+1); return [...m].map(([x,n])=>[x,String(n)]);}, edge:[] },
  { name:'record_total', param:'entries', type:'string[][]', out:'number', difficulty:'intermediate',
    instruction:'Each entry is [label, decimalInteger]. Return the sum of the second field. Inputs contain valid base-10 integers.',
    body:'{ let total = 0; for (const entry of entries) total += Number.parseInt(entry[1], 10); return total; }',
    make:r=>Array.from({length:1+Math.floor(r()*6)},(_,i)=>[`r${i}`,String(Math.floor(r()*31)-15)]), ref:a=>a.reduce((s,e)=>s+Number(e[1]),0), edge:[] },
  { name:'longest_word', param:'words', type:'string[]', out:'string', difficulty:'basic',
    instruction:'Return the longest string, keeping the earliest on ties. Length means JavaScript UTF-16 code units. Return empty string for no values.',
    body:'{ let best = ""; for (const word of words) if (word.length > best.length) best = word; return best; }',
    make:r=>stringArray(r,Math.floor(r()*7)), ref:a=>a.reduce((b,x)=>x.length>b.length?x:b,''), edge:[] },
  { name:'sorted_numbers', param:'numbers', type:'number[]', out:'number[]', difficulty:'intermediate',
    instruction:'Return a new array containing numbers in ascending numeric order. Do not mutate the input.',
    body:'{ return [...numbers].sort((a, b) => a - b); }',
    make:r=>ints(r,2+Math.floor(r()*7)), ref:a=>a.slice().sort((a,b)=>a-b), edge:[] },
  { name:'sort_pairs_by_key', param:'pairs', type:'string[][]', out:'string[][]', difficulty:'intermediate',
    instruction:'Sort [key, value] pairs by key in ascending JavaScript UTF-16 lexicographic order; preserve original order when keys tie. Return a new array.',
    body:'{ return pairs.slice().sort((a, b) => a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0); }',
    make:r=>Array.from({length:2+Math.floor(r()*6)},()=>[choose(r,['b','a','東京','é','a']),choose(r,words)]), ref:a=>a.map((pair,index)=>({pair,index})).sort((a,b)=>a.pair[0]<b.pair[0]?-1:a.pair[0]>b.pair[0]?1:a.index-b.index).map(x=>x.pair), edge:[] },
  { name:'parse_decimal', param:'text', type:'string', out:'number', difficulty:'intermediate',
    instruction:'Parse text as a signed base-10 integer after trimming surrounding whitespace. Return 0 when it is not a complete valid integer or the result is outside JavaScript safe-integer range; accept optional + or - and digits only.',
    body:'{ const value = text.trim(); if (!/^[+-]?\\d+$/.test(value)) return 0; const number = Number(value); return Number.isSafeInteger(number) ? number : 0; }',
    make:r=>choose(r,['42',' -7 ','+003','12x','','9007199254740991','  +0']), ref:s=>{const v=s.trim(); if(!/^[+-]?\d+$/.test(v))return 0;const n=Number(v);return Number.isSafeInteger(n)?n:0;}, edge:'1.5' },
  { name:'validate_email_like', param:'text', type:'string', out:'boolean', difficulty:'intermediate',
    instruction:'Return true exactly when text has one or more ASCII letters, digits, dots, underscores, plus signs, or hyphens before one @, then one or more ASCII letters/digits/hyphens, a dot, and at least two ASCII letters. No whitespace is allowed.',
    body:'{ return /^[A-Za-z0-9._+-]+@[A-Za-z0-9-]+\\.[A-Za-z]{2,}$/.test(text); }',
    make:r=>choose(r,['a.b+q@example.com','x@y.io','bad@@x.com','élan@example.com','a@b.c','a b@c.com']), ref:s=>/^[A-Za-z0-9._+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}$/.test(s), edge:'' },
  { name:'flatten_one_level', param:'rows', type:'number[][]', out:'number[]', difficulty:'basic',
    instruction:'Concatenate the numbers from each inner array in order. Flatten exactly one level; empty rows contribute nothing.',
    body:'{ const result = []; for (const row of rows) for (const value of row) result.push(value); return result; }',
    make:r=>nested(r,2+Math.floor(r()*5)), ref:a=>a.reduce((out,row)=>out.concat(row),[]), edge:[] },
  { name:'row_totals', param:'rows', type:'number[][]', out:'number[]', difficulty:'intermediate',
    instruction:'Return the sum of each row of numbers, in row order. An empty row has sum 0; preserve the original row order.',
    body:'{ return rows.map(row => row.reduce((sum, value) => sum + value, 0)); }',
    make:r=>nested(r,2+Math.floor(r()*5)), ref:a=>a.map(row=>row.reduce((s,x)=>s+x,0)), edge:[[]] },
  { name:'transpose_rectangular', param:'rows', type:'number[][]', out:'number[][]', difficulty:'advanced',
    instruction:'Transpose a rectangular number matrix: output column j contains input column j. For zero rows return []; for rows of zero columns return []. Input rows have equal lengths.',
    body:'{ if (rows.length === 0 || rows[0].length === 0) return []; const result = []; for (let col = 0; col < rows[0].length; col++) result.push(rows.map(row => row[col])); return result; }',
    make:r=>{const w=Math.floor(r()*4);return Array.from({length:1+Math.floor(r()*4)},()=>ints(r,w,0,9));}, ref:a=>a.length===0||a[0].length===0?[]:Array.from({length:a[0].length},(_,j)=>a.map(row=>row[j])), edge:[] },
  { name:'second_largest_distinct', param:'numbers', type:'number[]', out:'number', difficulty:'advanced',
    instruction:'Return the second-largest distinct number. Return 0 if fewer than two distinct numbers are present.',
    body:'{ let largest = 0, second = 0, hasLargest = false, hasSecond = false; for (const value of numbers) { if (!hasLargest || value > largest) { second = largest; hasSecond = hasLargest; largest = value; hasLargest = true; } else if (value < largest && (!hasSecond || value > second)) { second = value; hasSecond = true; } } return hasSecond ? second : 0; }',
    make:r=>ints(r,2+Math.floor(r()*7),-5,10), ref:a=>{const u=[...new Set(a)].sort((x,y)=>y-x);return u.length>1?u[1]:0;}, edge:[] },
  { name:'unicode_codepoint_count', param:'text', type:'string', out:'number', difficulty:'intermediate',
    instruction:'Count Unicode code points in text using JavaScript string iteration (a surrogate pair counts as one).',
    body:'{ let count = 0; for (const _character of text) count++; return count; }',
    make:r=>choose(r,['','A','éclair','🪁🪁','東京🌸']), ref:s=>Array.from(s).length, edge:'𠮷' },
  { name:'dedupe_sorted_numbers', param:'numbers', type:'number[]', out:'number[]', difficulty:'intermediate',
    instruction:'Return ascending numeric values with duplicates removed. Do not mutate numbers.',
    body:'{ const sorted = numbers.slice().sort((a, b) => a - b); return sorted.filter((value, index) => index === 0 || value !== sorted[index - 1]); }',
    make:r=>ints(r,3+Math.floor(r()*7),-3,5), ref:a=>[...new Set(a)].sort((a,b)=>a-b), edge:[] },
];

export function makeSyntheticCases(family, args) {
  const values = [args, family.edge];
  return values.map(value => {
    const expected = family.ref(value), checks = checkSyntheticProperties(family.name, value, expected);
    if (checks.some(check => !check.passed)) throw new Error(`reference property failed for ${family.name}`);
    return { args:[value], expected, outcome:'return', portable:true, property_checks:checks };
  });
}

// Structural/semantic checks complement exact reference replay. They are
// calculated on reference outputs; native replay acceptance then proves the
// candidate returned those same outputs for these cases.
export function checkSyntheticProperties(name, input, output) {
  const check = (property, passed) => ({ property, passed: Boolean(passed) });
  switch (name) {
    case 'stable_unique': return [check('unique values in first-occurrence order', Array.isArray(output) && output.every((x,i)=>output.indexOf(x)===i && input.indexOf(x)>=0 && (i===0 || input.indexOf(output[i-1])<input.indexOf(x))) && input.every(x=>output.includes(x)))];
    case 'prefix_sums': return [check('one cumulative total per input value', Array.isArray(output) && output.length===input.length && output.every((x,i)=>x===(i?output[i-1]:0)+input[i]))];
    case 'count_words': return [check('counts non-whitespace runs', output===(input.match(/\S+/gu)??[]).length)];
    case 'group_count': return [check('distinct first-seen keys with exact counts', Array.isArray(output) && output.length===[...new Set(input)].length && output.every(([key,count],i)=>key===[...new Set(input)][i] && count===String(input.filter(x=>x===key).length)))];
    case 'record_total': return [check('sum equals all decimal amount fields', output===input.reduce((sum,row)=>sum+Number(row[1]),0))];
    case 'longest_word': { const max=Math.max(0,...input.map(x=>x.length)); return [check('maximum UTF-16 length and earliest tie', output===(input.find(x=>x.length===max)??''))]; }
    case 'sorted_numbers': return [check('ascending order and preserved multiset', Array.isArray(output) && output.length===input.length && output.every((x,i)=>x===(i?Math.max(output[i-1],x):x)) && input.every(x=>output.filter(y=>y===x).length===input.filter(y=>y===x).length))];
    case 'sort_pairs_by_key': {
      const seen = new Map();
      const originalIndices = Array.isArray(output) ? output.map(pair => {
        const key = JSON.stringify(pair), occurrence = seen.get(key) ?? 0;
        seen.set(key, occurrence + 1);
        return input.flatMap((original, i) => JSON.stringify(original) === key ? [i] : [])[occurrence] ?? -1;
      }) : [];
      return [check('key order, preserved pairs, and stable ties', Array.isArray(output) && output.length===input.length
        && originalIndices.every(i=>i>=0) && output.every((pair,i)=>i===0 || output[i-1][0]<pair[0]
          || (output[i-1][0]===pair[0] && originalIndices[i-1]<originalIndices[i])))];
    }
    case 'parse_decimal': {
      const trimmed = input.trim(), parsed = Number(trimmed);
      const valid = /^[+-]?\d+$/.test(trimmed) && Number.isSafeInteger(parsed);
      return [check('safe integer or invalid-input sentinel', output === (valid ? parsed : 0))];
    }
    case 'validate_email_like': return [check('matches the documented ASCII pattern', output===/^[A-Za-z0-9._+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}$/.test(input))];
    case 'flatten_one_level': return [check('concatenates rows in order exactly once', Array.isArray(output) && JSON.stringify(output)===JSON.stringify(input.reduce((a,row)=>a.concat(row),[])))];
    case 'row_totals': return [check('one sum per row in order', Array.isArray(output) && output.length===input.length && output.every((sum,i)=>sum===input[i].reduce((a,x)=>a+x,0)))];
    case 'transpose_rectangular': return [check('rows and columns exchange positions', Array.isArray(output) && (input.length===0 || input[0].length===0 ? output.length===0 : output.length===input[0].length && output.every((row,j)=>row.length===input.length && row.every((x,i)=>x===input[i][j]))))];
    case 'second_largest_distinct': { const unique=input.filter((x,i)=>input.indexOf(x)===i).sort((a,b)=>b-a); return [check('second distinct maximum or empty sentinel', output===(unique.length>1?unique[1]:0))]; }
    case 'unicode_codepoint_count': return [check('counts Unicode code points', output===Array.from(input).length)];
    case 'dedupe_sorted_numbers': return [check('ascending unique values preserve input set', Array.isArray(output) && output.every((x,i)=>input.includes(x) && (i===0 || output[i-1]<x)) && output.length===[...new Set(input)].length)];
    default: throw new Error(`missing synthetic property checks for ${name}`);
  }
}
