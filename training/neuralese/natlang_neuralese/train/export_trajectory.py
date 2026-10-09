"""Export a complete recurrence checkpoint for inference and a new training stage.

The original optimizer/RNG checkpoint remains the authoritative resumable state.
Legacy recurrence checkpoints inherit frozen control rows and backbone metadata
from their pinned parent heads; this provenance is validated rather than guessed.
"""
import argparse, json, re
from pathlib import Path
import torch
from ..common.artifact_paths import ArtifactResolver
from ..common.hashing import sha256_file_hex as digest
from .trajectory_state import atomic_checkpoint

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--artifact',action='append',default=[],metavar='ROLE=PATH',
                   help='bind a recorded input (heads, pieces) to this file when the checkpoint cannot find it; its bytes must still match the pin')
    a=p.parse_args();state=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if state.get('schema')!='natlang.neuralese_recurrence_checkpoint/1':raise ValueError('unsupported checkpoint')
    identity=state['identity'];options=identity['options']
    artifacts=ArtifactResolver.from_state(state,overrides=dict(item.split('=',1) for item in a.artifact))
    parent_path=artifacts.path('heads');parent={}
    if parent_path:
        parent=torch.load(parent_path,map_location='cpu',weights_only=False,mmap=True)
    control=state.get('control_rows',parent.get('control_rows'));config=state.get('port_config',parent.get('port_config'))
    if control is None or not config:raise ValueError('missing deployment metadata; refuse reconstruction')
    pieces=Path(artifacts.path('pieces',required=True))
    texts=state.get('texts') or {r['name']:r['text'] for r in map(json.loads,pieces.open())}
    if a.out.exists():raise ValueError('fresh immutable export required')
    a.out.mkdir(parents=True)
    heads={**{k:parent[k] for k in ['backbone','lora_layers','lora_rank'] if k in parent},'heads':state['heads'],'lora':state['lora'],
      'control_rows':control,'port_config':config,'training_identity':identity}
    names=state.get('lora',{})
    if names:
        layers=sorted({int(m[1]) for name in names for m in [re.search(r'model\.layers\.(\d+)\.',name)] if m})
        ranks={int(value.shape[0]) for name,value in names.items() if '.lora_A.' in name}
        if len(ranks)>1:raise ValueError('mixed adapter ranks require an explicit deployment mapping')
        if layers and ranks:heads.update(lora_layers=layers,lora_rank=next(iter(ranks)))
    if state.get('control_head_rows',parent.get('control_head_rows')) is not None:
        heads['control_head_rows']=state.get('control_head_rows',parent.get('control_head_rows'))
    atomic_checkpoint(a.out/'heads.pt',heads)
    atomic_checkpoint(a.out/'soft-params.pt',{'params':state['params'],'texts':texts,'port_profile':config.get('profile','legacy-rms-v1')})
    receipt={'schema':'natlang.recurrence-deployment-export/1','checkpoint':str(a.checkpoint),'checkpoint_sha256':digest(a.checkpoint),'step':state['step'],
      'full_resume':'retain original optimizer/RNG checkpoint; these exports warm-start a new stage',
      'parent_heads':parent_path,'parent_heads_sha256':digest(parent_path) if parent_path else None,
      'sha256':{n:digest(a.out/n) for n in ['heads.pt','soft-params.pt']}}
    (a.out/'export.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
