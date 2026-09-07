"""Preference curation with group splits and explicit review provenance.

Synthetic/rule labels are not expert medical judgments. This module validates
the annotation contract; it cannot establish the clinical truth of an answer.
DPO reference: https://arxiv.org/abs/2305.18290.
"""
import hashlib
import unicodedata


def canonical(text):
    return ''.join(unicodedata.normalize('NFKC',text).lower().split())


def validate_pair(row):
    required={'id','group_id','prompt','chosen','rejected','reviewer_id','rubric_version',
              'chosen_safe','rejected_safe','source_ids','label_origin'}
    if required-row.keys():
        raise ValueError(f'missing preference fields: {sorted(required-row.keys())}')
    if any(not isinstance(row[k],str) or not row[k].strip() for k in
           ['id','group_id','prompt','chosen','rejected','reviewer_id','rubric_version']):
        raise ValueError('required text cannot be empty')
    if canonical(row['chosen'])==canonical(row['rejected']):
        raise ValueError('tied preference pair')
    if row['chosen_safe'] is not True or type(row['rejected_safe']) is not bool:
        raise ValueError('unsafe chosen or missing explicit safety judgment')
    if row['label_origin'] not in {'expert','model_judge','synthetic_fixture'}:
        raise ValueError('unrecognized label origin')
    if not isinstance(row['source_ids'],list) or not row['source_ids']:
        raise ValueError('knowledge provenance required')


def curate(rows,seed='medical-v1'):
    """Union patient/source groups linked by duplicate prompts BEFORE splitting.

    Hashing only prompt text can leak different conversations from one patient;
    hashing only patient IDs can leak duplicate prompts across patients. A union
    of both relations prevents those two leakage paths without reading labels.
    """
    if not rows: raise ValueError('empty dataset')
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x: parent[x]=find(parent[x])
        return parent[x]
    def union(a,b):
        a,b=find(a),find(b); parent[max(a,b)]=min(a,b)
    seen_ids=set()
    for row in rows:
        validate_pair(row)
        if row['id'] in seen_ids: raise ValueError('duplicate sample id')
        seen_ids.add(row['id'])
        union('group:'+row['group_id'],'prompt:'+canonical(row['prompt']))
    seen,out={}, {'train':[],'dev':[],'test':[]}
    for row in sorted(rows,key=lambda x:x['id']):
        key=canonical(row['prompt'])
        if key in seen:
            if seen[key]!=(canonical(row['chosen']),canonical(row['rejected'])):
                raise ValueError('conflicting labels for duplicate prompt; adjudication required')
            continue
        seen[key]=(canonical(row['chosen']),canonical(row['rejected']))
        group=find('group:'+row['group_id'])
        bucket=int(hashlib.sha256((seed+group).encode()).hexdigest()[:8],16)%100
        split='train' if bucket<80 else 'dev' if bucket<90 else 'test'
        out[split].append(dict(row,split=split,messages=[{'role':'user','content':row['prompt']}]))
    return out
