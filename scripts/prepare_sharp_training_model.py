"""Pin and verify trainable MiniCPM weights, retaining the selected Sharp tokenizer/template."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metadata',type=Path)
    parser.add_argument('sharp_tokenizer',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    metadata=json.loads(args.metadata.read_text())
    repository=metadata.get('id') or metadata['modelId']
    revision=metadata['sha']
    args.output.mkdir(parents=True,exist_ok=True)
    weights=[row for row in metadata['siblings'] if row['rfilename'].endswith('.safetensors')]
    if not weights:raise ValueError('No reviewed weight files')
    for row in metadata['siblings']:
        name=row['rfilename']
        if not name.endswith(('.safetensors','.json','.jinja','.model')):continue
        hf_hub_download(repository,name,revision=revision,local_dir=args.output)
    for row in weights:
        path=args.output/row['rfilename'];sha=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024**2),b''):sha.update(block)
        if path.stat().st_size!=row['size'] or sha.hexdigest()!=row['lfs']['sha256']:raise ValueError('Weight verification failed')
    tokenizer_files={}
    for name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja','generation_config.json'):
        source=args.sharp_tokenizer/name
        if not source.is_file():raise ValueError('Selected Sharp tokenizer file missing: '+name)
        shutil.copy2(source,args.output/name)
        tokenizer_files[name]=hashlib.sha256(source.read_bytes()).hexdigest()
    receipt={'repository':repository,'revision':revision,'weights':weights,'sharp_tokenizer_sha256':tokenizer_files,'format':'BF16 source weights; Sharp tokenizer and template; adapter-only QLoRA training'}
    (args.output/'natlang-model.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'ready':True,'repository':repository,'revision':revision,'output':str(args.output)}),flush=True)


if __name__=='__main__':main()
