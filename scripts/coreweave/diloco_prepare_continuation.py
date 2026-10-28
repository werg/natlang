"""Install a new plan and quarantine only inherited roles after checking the earned parent."""
import hashlib,json,os,sys
from pathlib import Path
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
from natlang_neuralese.common.jsonio import write_json_durable
root=Path(os.environ.get('NATLANG_CW_WORK','/work')); pin_path=Path(sys.argv[1]); pin_sha=sys.argv[2];plan_template=Path(sys.argv[3])
if hashlib.sha256(pin_path.read_bytes()).hexdigest()!=pin_sha:raise ValueError('earned pin binding differs')
pin=json.loads(pin_path.read_text());parent=Path(pin['path']);expected=pin['sha256'];step=pin['global_step'];size=pin['bytes']
if not str(parent).startswith(str(root)+'/') or step<1 or not pin.get('full_optimizer_rng_exposure'):
 raise ValueError('invalid earned parent pin')
instance=os.environ['NATLANG_CW_INSTANCE']; control=root/'.coreweave/instances'/instance
quarantine=control/'inherited-previous-run';marker=control/'earned-plan-ready.json'
if hashlib.file_digest(parent.open('rb'),'sha256').hexdigest()!=expected or parent.stat().st_size!=size:
 raise ValueError('earned full parent bytes differ from the exact earned pin')
receipt_path=Path(str(parent)+'.seed.json');receipt=json.loads(receipt_path.read_text())
if receipt['sha256']!=expected or receipt['bytes']!=parent.stat().st_size or receipt['global_step']!=step or not receipt['full_optimizer_rng_exposure']:
 raise ValueError('earned parent receipt differs')
plan=json.loads(plan_template.read_text()); dest=root/'diloco-resident/recipe-plan.json'
plan['handoffs'][-1]['seed_checkpoint']={'path':str(parent),'sha256':expected,'bytes':receipt['bytes'],'global_step':step}
plan['handoffs'][-1]['seed_receipt']={'path':str(receipt_path),'sha256':hashlib.sha256(receipt_path.read_bytes()).hexdigest()}
if marker.exists():
 if json.loads(marker.read_text())!={'parent_sha256':expected,'plan':plan}:raise ValueError('existing handoff marker differs')
 if not dest.is_file() or json.loads(dest.read_text())!=plan:raise ValueError('installed plan differs from handoff')
else:
 state=CheckpointPolicy.load(parent);training=state['training'];old=json.loads(training['options']['recipe_binding'])
 if training['global_step']!=step or old.get('lineage_sha256',old['sha256'])!=plan['recipe_sha256'] or not training.get('optimizer') or not training.get('rng') or not training.get('exposure_counts') or not training.get('ema'):
  raise ValueError('earned parent lacks full earned state or recipe lineage')
 del state,training
 quarantine.mkdir(parents=True,exist_ok=True)
 phase=control/'earned-quarantine-complete.json'
 if phase.exists():
  if json.loads(phase.read_text())!={'parent_sha256':expected}:raise ValueError('quarantine journal differs')
 else:
  for name in ('diloco-coordinator','diloco-learner','diloco-resident'):
   source=root/name;target=quarantine/name
   if source.is_symlink():raise ValueError('inherited role state is a symlink')
   if source.exists():
    if target.exists():raise ValueError('both inherited role and quarantine already exist')
    os.rename(source,target)
  write_json_durable(phase,{'parent_sha256':expected})
 if dest.exists() and json.loads(dest.read_text())!=plan:raise ValueError('existing earned plan differs')
 if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);write_json_durable(dest,plan)
 write_json_durable(marker,{'parent_sha256':expected,'plan':plan})
print(json.dumps({'event':'earned-parent-handoff-ready','sha256':expected,'global_step':step,'plan':str(dest)}))
