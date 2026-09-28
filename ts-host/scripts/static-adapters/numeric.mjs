import { adaptedCase, requireValue, evalCall, returnCall } from './common.mjs';

export function parseFinqaProgram(source) {
  requireValue(typeof source === 'string', 'missing_original_numeric_program');
  const steps = [], values = [];
  let remainder = source.trim();
  for (let index = 0; remainder && index < 64; index++) {
    const match = /^(add|subtract|multiply|divide|exp|greater)\(\s*([^(),]+)\s*,\s*([^(),]+)\s*\)/.exec(remainder);
    requireValue(match, 'unsupported_numeric_program');
    const operands = match.slice(2).map(raw => {
      const token = raw.trim();
      if (/^#\d+$/.test(token)) {
        const at = Number(token.slice(1)); requireValue(at < values.length, 'invalid_numeric_reference');
        requireValue(typeof values[at] === 'number', 'numeric_reference_to_text');
        return {kind:'result', index:at};
      }
      const text = token.replace(/^const_/, '');
      requireValue(/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?$/.test(text), 'invalid_numeric_literal');
      const value = Number(text.replace(/%$/, '')) / (text.endsWith('%') ? 100 : 1);
      requireValue(Number.isFinite(value), 'nonfinite_numeric_literal');
      return {kind:'literal', value};
    });
    const [a,b] = operands.map(o=>o.kind==='result'?values[o.index]:o.value);
    requireValue(match[1] !== 'divide' || b !== 0, 'division_by_zero');
    const computed = ({add:()=>a+b, subtract:()=>a-b, multiply:()=>a*b, divide:()=>a/b,
      exp:()=>a**b, greater:()=>a>b?'yes':'no'})[match[1]]();
    requireValue(typeof computed === 'string' || Number.isFinite(computed), 'nonfinite_numeric_result');
    steps.push({op:match[1], operands}); values.push(computed);
    remainder = remainder.slice(match[0].length).trim();
    if (remainder) { requireValue(remainder.startsWith(','), 'missing_numeric_separator'); remainder=remainder.slice(1).trim(); requireValue(remainder, 'trailing_numeric_separator'); }
  }
  requireValue(steps.length && !remainder, 'numeric_program_too_long');
  return {steps, values, answer:values.at(-1)};
}

export function finqa(row, info, split='train') {
  const qa = row.qa;
  requireValue(qa && typeof qa.question === 'string' && qa.question.trim(), 'missing_numeric_question');
  const parsed = parseFinqaProgram(qa.program);
  const gold = typeof parsed.answer === 'number' ? Number(qa.exe_ans) : qa.exe_ans;
  requireValue(typeof gold === typeof parsed.answer && gold === parsed.answer, 'numeric_source_answer_mismatch');
  // Full source text/table avoids quietly turning gold_inds into the only evidence.
  const evidence = [...(row.pre_text ?? []), 'Table:', ...(row.table ?? []).map(cells=>cells.join(' | ')), ...(row.post_text ?? [])].join('\n');
  requireValue(evidence.trim(), 'missing_numeric_evidence');
  requireValue(evidence.length < 24000, 'oversize_numeric_evidence');
  const original = {id:`finqa:${split}:${row.id}`, split, license:'CC-BY-4.0', source_ids:[row.id],
    source_groups:[row.id, `document:${String(row.id).replace(/-\d+$/, '')}`],
    source_revisions:[info.sha256], gold_sources:['FinQA:original-executable-program']};
  const actions = parsed.steps.map((step,index)=> {
    const [a,b] = step.operands.map(o=>o.kind==='result'?`step_${o.index}`:JSON.stringify(o.value));
    const expression = ({add:`(${a}+${b})`, subtract:`(${a}-${b})`, multiply:`(${a}*${b})`,
      divide:`(${a}/${b})`, exp:`Math.pow(${a},${b})`, greater:`(${a}>${b}?'yes':'no')`})[step.op];
    return evalCall(index===parsed.steps.length-1 ? `return ${expression};` : `const step_${index} = ${expression};`);
  });
  const record = adaptedCase(original, info, {family:'finqa',
    root:{name:'answer_financial_question', args:{question:'string', evidence:'string'}, returns:typeof gold,
      instructions:'Read the question and supplied report evidence. Compute the requested answer with exact arithmetic, preserving intermediate values. Return the computed number, or yes/no for a comparison. Do not infer facts outside the report.'},
    inputs:{question:qa.question, evidence}, expected:gold, actions:[...actions,returnCall(gold)],
    quality:'held', checks:['question_units_and_display_answer_require_independent_review','original_program_parsed_with_typed_operands','exact_executable_answer_agreement','full_report_evidence_retained']});
  record.external_source.original_row = row;
  record.numeric_program = {source:qa.program, steps:parsed.steps};
  return record;
}
