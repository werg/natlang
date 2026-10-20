"""Authenticate completed Search rendering in its exact historical import tree."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

FILES = {
 'scripts/neuralese_data/render_native_turn_corpus.py': '69bae25524c06c2806afbdd1175e8e376e52b342e2544def38dbc2d489288658',
 'training/neuralese/natlang_neuralese/data/text_corpus.py': '622b4ad08257ed09d0dd10d52d97adfa1c45cc8b9c8c46bbeef57fae6caff63b',
 'scripts/neuralese_data/strip_generic_system_prompt.py': 'ebedd490684608188473f19eaa214d7be39dbf762a57d20e8955893741727838',
 'scripts/neuralese_data/conversion_output_lease.py': 'a7b47b5bd32afc0bf74f9de82f3e0656e541acb8f6eb28ec88b5dacaab85258c',
}
PARENT = 'external-expansion-search-owl-native-20261015-v1'
MODELS = ('lfm25','mellum21')
_cached_sha = None

def cached_hashes():
 global _cached_sha
 from natlang_neuralese.common.hashing import sha256_file_hex_cached
 _cached_sha=sha256_file_hex_cached


def binding(path):
 path=Path(path)
 if _cached_sha is None:
  with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
 else:digest=_cached_sha(path,trust_archive_receipt=False)
 return dict(path=str(path),sha256=digest,bytes=path.stat().st_size)


def check(reference):
 actual=binding(reference['path'])
 if any(actual[key]!=reference[key] for key in ('sha256','bytes') if key in reference):
  raise ValueError('completed Search verifier input changed')
 return Path(reference['path'])


def composition(path):
 ref=binding(path);value=json.loads(Path(path).read_bytes());tree=Path(value['tree']).resolve(strict=True)
 if set(value['files'])!=set(FILES):raise ValueError('historical Search composition inventory differs')
 for relative,sha in FILES.items():
  item=value['files'][relative]
  if Path(item['path'])!=tree/relative or item['sha256']!=sha:raise ValueError('unqualified historical Search dependency')
  check(item)
 return ref,value,tree


def verified_artifacts(receipt, *, shard, launch_path, composition_path):
 """The current converter must consume exactly the source the child verified."""
 if receipt.get('schema')!='natlang.search-completed-parent-verification/1' or receipt.get('passed')is not True:
  raise ValueError('historical Search verification did not complete')
 if receipt.get('shard')!=shard or receipt.get('launch')!=binding(launch_path) or receipt.get('composition')!=binding(composition_path):
  raise ValueError('historical Search verification names another job')
 composition(composition_path)
 if receipt.get('verifier')!=binding(Path(__file__).resolve()):raise ValueError('historical Search verifier source changed')
 artifacts=receipt.get('artifacts')
 if not isinstance(artifacts,list)or not artifacts or len({x['path']for x in artifacts})!=len(artifacts):
  raise ValueError('historical Search verification lost artifact inventory')
 for item in artifacts:check(item)
 return receipt


def verify(shard, *, launch_path, composition_path, output):
 """Use a new interpreter; never impersonate historical imports in this process."""
 cached_hashes();composition(composition_path)
 command=[sys.executable,str(Path(__file__).resolve()),'--worker','--shard',shard,
          '--launch',str(launch_path),'--composition',str(composition_path),'--output',str(output)]
 subprocess.run(command,check=True)
 receipt=json.loads(Path(output).read_bytes())
 return verified_artifacts(receipt,shard=shard,launch_path=launch_path,composition_path=composition_path)


def parent_job_binding(renderer, job, done, selected):
 parent=done.get('renderer_code_sha256')
 if parent not in renderer.SEARCH_REVIEWED_RESOURCE_PARENT_CHAIN:
  raise ValueError('unreviewed Search renderer parent')
 expected=renderer._render_job_binding_from_job(job,renderer.file_sha(selected),renderer._resource_parent_code_binding(parent,job))
 if done.get('render_job_binding')!=expected:raise ValueError('historical Search job equation differs')
 return expected


def worker(shard, *, launch_path, composition_path, output):
 if not shard.isdigit()or len(shard)!=4:raise ValueError('Search shard must have its exact four-digit identity')
 original_composition,description,tree=composition(composition_path)
 # No natlang module was imported before selecting this authenticated tree.
 if any(key=='natlang_neuralese'or key.startswith('natlang_neuralese.')for key in sys.modules):
  raise ValueError('historical Search verifier requires a fresh interpreter')
 sys.path[:0]=[str(tree/'training/neuralese'),str(tree)]
 from scripts.neuralese_data import render_native_turn_corpus as renderer
 cached_hashes()
 if Path(renderer.__file__).resolve()!=tree/'scripts/neuralese_data/render_native_turn_corpus.py':
  raise ValueError('historical Search renderer import escaped its composition')
 launch_ref=binding(launch_path);launch=json.loads(check(launch_ref).read_bytes());work=Path(launch['out_dir'])/'work'
 if not work.is_absolute():work=Path(launch['cwd'])/work if 'cwd'in launch else Path(__file__).resolve().parents[2]/work
 state_ref=binding(work/'selection-state.json');state=json.loads(check(state_ref).read_bytes())
 if state['next_source']!=len(launch['source_turn_files']):raise ValueError('Search source selection remains incomplete')
 source_manifest=binding(launch['corpus_manifest'])
 if source_manifest['sha256']!=launch['corpus_manifest_sha256']:raise ValueError('Search source manifest changed')
 manifest=json.loads(check(source_manifest).read_bytes())
 if any(Path(row['path']).name in('native-pieces.jsonl','pieces.jsonl')for row in manifest['files']):
  raise ValueError('Search historical packet has undeclared piece sidecars')
 tokenizers=dict(item.split('=',1)for item in launch['tokenizers'])
 if set(tokenizers)!=set(MODELS):raise ValueError('Search student inventory differs')
 renderer._init(tokenizers,{})
 input_binding=renderer._render_input_binding(tokenizers,{})
 held=set(state['test_groups'])|set(state['reserved_groups'])|renderer.held_groups_from([Path(launch['held_text'])])
 artifacts={}
 def retain(path):
  item=binding(path);prior=artifacts.setdefault(item['path'],item)
  if prior!=item:raise ValueError('Search artifact changed during verification')
 for item in(original_composition,launch_ref,state_ref,source_manifest):retain(item['path'])
 retain(launch['held_text']);selected=work/'selected'/f'{shard}.jsonl.zst';retain(selected)
 parents={}
 for alias in MODELS:
  marker=work/'done'/alias/f'{shard}.json';retain(marker);done=json.loads(marker.read_bytes());parent=done.get('renderer_code_sha256')
  job=(str(selected),str(work),alias,state['blocked'],PARENT,held,set(),False,set(),tuple(launch['renderings']),True,set(),None,input_binding,None)
  expected=parent_job_binding(renderer,job,done,selected)
  renderer._verify_trajectory_receipts(job,done,set(renderer.SEARCH_REVIEWED_RESOURCE_PARENT_CHAIN)|{renderer.LEGACY_SEARCH_RENDERER_SHA256,FILES['scripts/neuralese_data/render_native_turn_corpus.py']},expected_job_binding=expected)
  for key in('decontaminated','decision_excluded','target_excluded'):
   if done.get(key)!=0:raise ValueError('Search exclusions require explicit admission decision')
  for relative in done['aggregate_outputs']:retain(work/relative)
  receipts=work/'trajectory-stream'/alias/shard/'receipts'
  for path in sorted(receipts.glob('*.json')):
   retain(path);value=json.loads(path.read_bytes());group=path.parent.parent/'groups'/path.stem
   for relative in value['outputs']:retain(group/relative)
  parents[alias]=dict(renderer=parent,job_binding=expected,marker=binding(marker),records=done['records'])
 for item in artifacts.values():check(item)
 if composition(composition_path)[0]!=original_composition:raise ValueError('historical Search composition changed')
 result=dict(schema='natlang.search-completed-parent-verification/1',passed=True,verifier=binding(Path(__file__).resolve()),shard=shard,launch=launch_ref,composition=original_composition,parents=parents,artifacts=list(artifacts.values()),source_row_parses=0,source_render_calls=0,training_admission=False,new_earned_visits=0)
 p=Path(output);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if json.loads(p.read_bytes())!=result:raise ValueError('historical Search verification output already differs')
 else:
  with p.open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
 return result


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true',required=True);parser.add_argument('--shard',required=True);parser.add_argument('--launch',type=Path,required=True);parser.add_argument('--composition',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 worker(args.shard,launch_path=args.launch,composition_path=args.composition,output=args.output)


if __name__=='__main__':main()
