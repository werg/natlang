"""Export a complete recurrence checkpoint for inference and a new training stage.

The original optimizer/RNG checkpoint remains the authoritative resumable state.
Legacy recurrence checkpoints inherit frozen control rows and backbone metadata
from their pinned parent heads; this provenance is validated rather than guessed.
"""
import argparse, hashlib, json, re
from pathlib import Path
import torch
from .trajectory_state import atomic_checkpoint

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();state=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if state.get('schema')!='natlang.neuralese_recurrence_checkpoint/1':raise ValueError('unsupported checkpoint')
    identity=state['identity'];options=identity['options'];parent_path=options.get('heads');parent={}
    if parent_path:
        if digest(parent_path)!=identity['files'][str(Path(parent_path).resolve())]:raise ValueError('parent heads changed')
        parent=torch.load(parent_path,map_location='cpu',weights_only=False,mmap=True)
    control=state.get('control_rows',parent.get('control_rows'));config=state.get('port_config',parent.get('port_config'))
    if control is None or not config:raise ValueError('missing deployment metadata; refuse reconstruction')
    pieces=Path(options['pieces'])
    if digest(pieces)!=identity['files'][str(pieces.resolve())]:raise ValueError('piece initialization changed')
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
    atomic_checkpoint(a.out/'heads.pt',heads)
    atomic_checkpoint(a.out/'soft-params.pt',{'params':state['params'],'texts':texts})
    receipt={'schema':'natlang.recurrence-deployment-export/1','checkpoint':str(a.checkpoint),'checkpoint_sha256':digest(a.checkpoint),'step':state['step'],
      'full_resume':'retain original optimizer/RNG checkpoint; these exports warm-start a new stage',
      'parent_heads':parent_path,'parent_heads_sha256':digest(parent_path) if parent_path else None,
      'sha256':{n:digest(a.out/n) for n in ['heads.pt','soft-params.pt']}}
    (a.out/'export.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
