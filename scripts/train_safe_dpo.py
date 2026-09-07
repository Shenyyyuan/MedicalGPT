"""Medical preference alignment with an explicit safety-reviewed data contract.

Uses the auditable DPO core vendored in src/alignment. Existing TRL trainers in
training/ remain available as upstream alternatives. This is preference filtering,
not a formal guarantee that the trained policy satisfies a clinical constraint.
"""
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
import torch
from src.medical_alignment.data import validate_pair,canonical
from src.alignment.models import load_actor,encode_pair
from src.alignment.trainers import train_dpo_batch


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): result.update(chunk)
    return result.hexdigest()


def run(config):
    def read(path,split):
        rows=[json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
        if not rows: raise ValueError('nonempty train and dev splits required')
        for r in rows:
            validate_pair(r)
            if r.get('split')!=split: raise ValueError('split mismatch')
            if r['label_origin']=='synthetic_fixture': raise ValueError('fixtures are for software tests, not medical training')
        return rows
    train,dev=read(config['train'],'train'),read(config['dev'],'dev')
    if ({r['group_id'] for r in train}&{r['group_id'] for r in dev} or
        {canonical(r['prompt']) for r in train}&{canonical(r['prompt']) for r in dev}):
        raise ValueError('train/dev contamination')
    output=Path(config['output'])
    if output.exists() and any(output.iterdir()): raise FileExistsError('use a new run directory')
    torch.manual_seed(config.get('seed',42)); random.seed(config.get('seed',42))
    policy,tokenizer=load_actor(config['checkpoint'],config.get('device','cpu'),config.get('lora_rank',8))
    reference=copy.deepcopy(policy).requires_grad_(False).eval()
    def encode(rows):
        pairs=[]
        for r in rows:
            prompt=[{'role':'user','content':r['prompt']}]
            c,n=encode_pair(tokenizer,prompt,r['chosen'],config.get('max_length',2048))
            bad,m=encode_pair(tokenizer,prompt,r['rejected'],config.get('max_length',2048))
            if n!=m: raise ValueError('preference prompt mismatch')
            pairs.append((c,bad,n))
        return pairs
    pairs,dev_pairs=encode(train),encode(dev)
    optimizer=torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],lr=config.get('learning_rate',1e-6))
    logs=[]
    for epoch in range(config.get('epochs',1)):
        random.shuffle(pairs)
        for pair in pairs: logs.append(dict(train_dpo_batch(policy,reference,optimizer,[pair],config.get('beta',.1)),epoch=epoch))
    device=next(policy.parameters()).device
    margins=[]
    with torch.no_grad():
        for c,r,n in dev_pairs:
            pc=policy.score(c.to(device),n)[0].sum(); pr=policy.score(r.to(device),n)[0].sum()
            rc=reference.score(c.to(device),n)[0].sum(); rr=reference.score(r.to(device),n)[0].sum()
            margins.append(float(pc-pr-rc+rr))
    output.mkdir(parents=True,exist_ok=True)
    policy.actor.save_pretrained(output/'policy'); tokenizer.save_pretrained(output/'policy')
    torch.save(optimizer.state_dict(),output/'optimizer.pt')
    report={'scope':'preference_training_not_clinical_evaluation','config':config,'train_n':len(train),'dev_n':len(dev),
        'dev_preference_accuracy':sum(m>0 for m in margins)/len(margins),'logs':logs,
        'input_sha256':{k:digest(config[k]) for k in ['train','dev']},
        'checkpoint_files':{p.name:digest(p) for p in Path(config['checkpoint']).iterdir()
            if p.is_file() and p.suffix in {'.json','.safetensors','.bin'}}}
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); args=p.parse_args()
    run(json.loads(Path(args.config).read_text(encoding='utf-8')))
