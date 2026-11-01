#!/usr/bin/env python3
"""Compile a complete authenticated Luna episode into a native training graph, without model execution."""
import argparse,json,os,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'training/neuralese'))

def main():
 from scripts.neuralese_data.provider_execution_runtime import require_frozen_provider_runtime
 identity=require_frozen_provider_runtime(ROOT)
 from transformers import AutoTokenizer
 from natlang_neuralese.data.provider_teacher_source import _read_bound
 from natlang_neuralese.data.provider_static_graph import compile_verified_episode
 from natlang_neuralese.data.text_corpus import tokenizer_fingerprint
 from natlang_neuralese.model.shared_state import RoleTokens,provision_role_tokenizer
 from natlang_neuralese.serve.engine import Engine
 from natlang_neuralese.serve.chat import bind_history_reasoning,tool_call_format
 from natlang_neuralese.common.hashing import sha256_hex
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',required=True);p.add_argument('--base',required=True);p.add_argument('--output',required=True);p.add_argument('--tokenizer-fingerprint',required=True);a=p.parse_args()
 audit=Path(a.audit).resolve(strict=True);record=json.loads(audit.read_bytes());ref=record['python_corpus_conversion'];rows=[json.loads(line) for line in _read_bound(ref['path'],ref['sha256']).splitlines() if line.strip()];ids=[r['id'] for r in rows]
 if len(ids)!=len(set(ids)) or not ids:raise ValueError('source decision IDs missing or duplicate')
 base=Path(a.base).resolve(strict=True);config=json.loads((base/'config.json').read_text());model_type=config.get('model_type')
 if model_type!='lfm3':raise ValueError('static compiler currently qualifies only the exact LFM3 tokenizer contract')
 t=AutoTokenizer.from_pretrained(str(base),local_files_only=True,trust_remote_code=False); roles=RoleTokens(*provision_role_tokenizer(t));fingerprint=tokenizer_fingerprint(t)
 if fingerprint!=a.tokenizer_fingerprint:raise ValueError('tokenizer fingerprint differs from declared target')
 t=bind_history_reasoning(t,policy='keep',model_type=model_type);renderer=Engine.__new__(Engine);renderer.tokenizer=t;renderer.role_tokens=roles;renderer.stop_ids={i for i in (t.convert_tokens_to_ids('<|im_end|>'),t.eos_token_id) if i is not None};renderer.tool_call_format=tool_call_format(model_type);renderer.prompt_rendering='full'
 graph,example=compile_verified_episode(str(audit),ids,renderer,tokenizer_fingerprint=fingerprint);graph['static_training_provenance']['execution_runtime']=identity
 output=Path(a.output).resolve();output.mkdir(parents=True,exist_ok=False);dest=output/'native-episode.jsonl';payload=(json.dumps(graph,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode();dest.write_bytes(payload)
 receipt={'schema':'natlang.luna-static-episode-build/1','audit':{'path':str(audit),'sha256':sha256_hex(audit.read_bytes())},'graph':{'path':str(dest),'sha256':sha256_hex(payload)},'runtime':identity,'source_turn_count':len(ids),'nodes':len(graph['nodes']),'supervised_tokens':sum(w>0 for w in example.target_weights),'imports':sum(len(n.imports) for n in example.plan.invocations),'scope':'Authenticated teacher targets and native graph topology; no model forward, KV capture or sampled policy probabilities.','training_admission':False}
 (output/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps({'graph':str(dest),'receipt':str(output/'receipt.json'),'nodes':receipt['nodes'],'supervised_tokens':receipt['supervised_tokens'],'imports':receipt['imports']}),flush=True)
if __name__=='__main__':main()
