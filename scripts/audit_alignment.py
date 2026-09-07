"""Offline review aggregation / preference curation. Never calls a judge API."""
import argparse
import hashlib
import json
from pathlib import Path
from src.medical_alignment.data import curate
from src.medical_alignment.evaluation import aggregate,paired_compare


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['curate','evaluate'])
    parser.add_argument('--input',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(); inp=Path(args.input)
    rows=[json.loads(line) for line in inp.read_text(encoding='utf-8').splitlines() if line.strip()]
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    if args.mode=='curate':
        result=curate(rows)
        for split,items in result.items():
            (output/f'{split}.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in items),encoding='utf-8')
        report={'counts':{k:len(v) for k,v in result.items()},'label_origins':sorted({r['label_origin'] for r in rows})}
    else:
        models={name:[r for r in rows if r['model']==name] for name in sorted({r['model'] for r in rows})}
        report={'models':{name:aggregate(items) for name,items in models.items()}}
        if 'SFT' in models:
            report['vs_sft']={name:paired_compare(models['SFT'],items) for name,items in models.items() if name!='SFT'}
        report['scope']='synthetic_fixture' if all(r.get('synthetic_fixture') is True for r in rows) else 'provided_reviewer_annotations'
    report['input_sha256']=hashlib.sha256(inp.read_bytes()).hexdigest()
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
