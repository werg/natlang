#!/usr/bin/env python3
"""Decode bounded, deterministic effective-target samples for human review."""
import argparse
from collections import Counter
import hashlib
import json
import re
from pathlib import Path

from natlang_neuralese.common.jsonio import write_json_durable
from scripts.neuralese_data.build_acquired_streaming_successor import bind, checked, read
from scripts.neuralese_data.student_tokenizers import resolve

CODE = Path(__file__).read_bytes()
CALL = re.compile(r'<tool_call>(.*?)</tool_call>', re.S)
FUNCTION = re.compile(r'\s*<function=([A-Za-z_][\w.:-]*)>(.*?)</function>\s*', re.S)
PARAMETER = re.compile(r'<parameter=([A-Za-z_][\w.:-]*)>(.*?)</parameter>', re.S)


def assess_call(text):
    """Check visible markup structure only; do not judge API correctness or execution."""
    match = FUNCTION.fullmatch(text)
    if not match: return {'well_formed_markup': False, 'function': None}
    body = match.group(2)
    leftovers = PARAMETER.sub('', body)
    return {'well_formed_markup': not leftovers.strip(), 'function': match.group(1)}


def review(census_path, output):
    census_path = Path(census_path).resolve(strict=True); census_ref = bind(census_path)
    census = read(census_path)
    if census.get('schema') != 'natlang.acquired-effective-learning-census/1' or census.get('training_admission') is not False:
        raise ValueError('requires a completed unadmitted source census')
    from transformers import AutoTokenizer
    student = resolve('lfm3'); tokenizer_path = Path(student['path'])
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
    output = Path(output)
    if output.exists(): raise ValueError('sample review requires a fresh output directory')
    output.mkdir(parents=True)
    samples = []; counts = Counter(); content_hashes = Counter(); functions = Counter()
    idioms = Counter(); source_stores = Counter(); sample_row_mass = Counter()
    for proof_ref in census['leaf_index_proofs']:
        proof_path = checked(proof_ref, 'leaf parser proof')
        leaf_path = proof_path.parent.parent / 'completed.json'
        leaf = read(leaf_path)
        if leaf.get('index_proof') != proof_ref: raise ValueError('leaf completion does not bind the census proof')
        reviews = checked(leaf['reviews'], 'leaf content review')
        signals_path = checked(leaf['signals'], 'leaf effective-loss signals')
        signals = {}
        with signals_path.open() as stream:
            for line in stream:
                signal = json.loads(line)
                if signal['id'] in signals: raise ValueError('leaf loss signals repeat an identity')
                signals[signal['id']] = signal
        with reviews.open() as source:
            for line in source:
                item = json.loads(line); sample = item.get('sample')
                if sample is None: continue
                ident = item['id']
                signal = signals.get(ident)
                if signal is None: raise ValueError('sample lacks its effective-loss row')
                source_stores[signal.get('source_store') or '<unknown>'] += 1
                sample_row_mass.update(signal['category_mass'])
                if (sample.get('selection') != 'sha256(graph-id) modulo 512 equals zero'
                        or int(hashlib.sha256(ident.encode()).hexdigest()[:8],16) % 512
                        or sample.get('token_limit') != 8192):
                    raise ValueError('sample does not follow the declared deterministic selection')
                segments = sample['effective_positive_token_segments']
                if sum(map(len,segments)) > 8192: raise ValueError('sample exceeds its token bound')
                decoded = [tokenizer.decode(tokens, skip_special_tokens=False) for tokens in segments]
                calls = []
                for segment in decoded:
                    counts['opening_call_tags'] += segment.count('<tool_call>')
                    counts['closing_call_tags'] += segment.count('</tool_call>')
                    for match in CALL.finditer(segment):
                        result = assess_call(match.group(1));calls.append(result)
                        counts['complete_calls'] += 1
                        counts['complete_calls_markup_failed'] += not result['well_formed_markup']
                        if result['function']: functions[result['function']] += 1
                text = '\n'.join(decoded)
                for name, pattern in {
                    'await_nl_call': r'\bawait\s+nl\s*\(',
                    'nl_call': r'\bnl\s*\(', 'iterateOn': r'\biterateOn\s*\(',
                    'sort_call': r'\.sort\s*\(', 'finish_true': r'\bfinish\s*:\s*true\b',
                    'tool_eval': r'<function=eval>',
                }.items():
                    if re.search(pattern,text): idioms[name] += 1
                normalized = ' '.join(' '.join(decoded).split())
                content_hashes[hashlib.sha256(normalized.encode()).hexdigest()] += 1
                samples.append({'id':ident,'source':leaf['source'],'source_store':signal.get('source_store'),
                    'full_row_category_mass':signal['category_mass'],'segments':decoded,
                    'possibly_truncated':sample['possibly_truncated'],'complete_calls':calls})
    sample_path = output / 'decoded-supervised-samples.jsonl'
    with sample_path.open('x') as stream:
        for sample in samples: stream.write(json.dumps(sample,ensure_ascii=False,sort_keys=True)+'\n')
    core_native_tools = {'eval','return_result','read_code','read_page','compact_history','escalate'}
    agent_source = Path(__file__).resolve().parents[2]/'ts-host/src/native/agent.ts'
    summary = {'schema':'natlang.acquired-supervised-sample-review/1','training_admission':False,
        'census':census_ref,'rows_sampled':len(samples),'counts':dict(counts),
        'complete_call_functions':dict(functions),'sample_source_stores':dict(source_stores),
        'core_native_tools':sorted(core_native_tools),
        'sample_calls_outside_core_native_names':sum(n for name,n in functions.items() if name not in core_native_tools),
        'native_agent_source':bind(agent_source),
        'sampled_full_row_category_mass':dict(sample_row_mass),'sample_idiom_rows':dict(idioms),
        'normalized_sample_duplicate_excess':sum(n-1 for n in content_hashes.values() if n>1),
        'decoded_samples':bind(sample_path),
        'tokenizer':bind(tokenizer_path/'tokenizer.json'), 'chat_template':bind(tokenizer_path/'chat_template.jinja'),
        'builder':bind(Path(__file__)),
        'scope':'Deterministic sampled effective-positive spans, capped at 8192 tokens per row. Idiom and function counts cover sampled spans, not whole graphs. Category mass covers complete sampled rows, not full-corpus proportions. Markup checks do not validate tools, outcomes, or semantics; incomplete spans are not classified as malformed.'}
    if bind(census_path)!=census_ref or Path(__file__).read_bytes()!=CODE:
        raise ValueError('census or review program changed')
    write_json_durable(output/'receipt.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--census',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();print(json.dumps(review(args.census,args.out),sort_keys=True))


if __name__=='__main__':main()
