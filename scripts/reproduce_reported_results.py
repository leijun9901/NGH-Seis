"""Download, hash-verify and reevaluate the two reported IID reference models.

This downloads the 100-sample fixed IID TEST subset, not the full training data.
It evaluates the original weights; it does not claim to repeat training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import requests

PROJECT=Path(__file__).resolve().parents[1]

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def valid(path,info):
    return path.is_file() and path.stat().st_size==info['size_bytes'] and sha256(path)==info['sha256']

def download(url,path,info):
    if valid(path,info):return
    partial=path.with_suffix(path.suffix+'.partial')
    offset=partial.stat().st_size if partial.exists() else 0
    headers={'User-Agent':'NGH-Seis-reproducibility'}
    if offset:headers['Range']=f'bytes={offset}-'
    with requests.get(url,headers=headers,stream=True,timeout=(30,120)) as r:
        r.raise_for_status()
        mode='ab' if offset and r.status_code==206 else 'wb'
        with partial.open(mode) as f:
            for block in r.iter_content(4*1024*1024):
                if block:f.write(block)
    if not valid(partial,info):raise RuntimeError(f'Size/SHA-256 verification failed: {path.name}')
    partial.replace(path)

def extract_bundle(bundle,destination):
    root=destination.resolve()
    with zipfile.ZipFile(bundle) as z:
        for member in z.infolist():
            target=(root/member.filename).resolve()
            if not target.is_relative_to(root):raise ValueError('Unsafe archive member')
        z.extractall(root)
    release=root/'NGH-Seis_v1.0'
    manifest=json.loads((release/'benchmark_bundle_manifest.json').read_text(encoding='utf-8'))
    if len(manifest['candidate_ids'])!=100:raise ValueError('Expected 100 fixed IID test samples')
    split=json.loads((release/'metadata/iid_splits_80_10_10.json').read_text(encoding='utf-8'))
    if sorted(manifest['candidate_ids'])!=sorted(split['test']):raise ValueError('IID test IDs do not match')
    for info in manifest['files']:
        path=(release/info['path']).resolve()
        if not path.is_relative_to(release.resolve()) or not valid(path,info):
            raise RuntimeError(f'Bundle content verification failed: {info["path"]}')
    return release,dict(status='pass',test_samples=100,files_checked=len(manifest['files']),
                        scope=manifest['scope'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workdir',type=Path,default=PROJECT/'outputs/reproduction')
    p.add_argument('--task',choices=('1','2','both'),default='both')
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    p.add_argument('--assets-dir',type=Path,help='Use already downloaded release assets, still SHA-256 checked')
    p.add_argument('--verify-only',action='store_true')
    args=p.parse_args()
    config=json.loads((PROJECT/'metadata/benchmark_assets.json').read_text(encoding='utf-8'))
    args.workdir.mkdir(parents=True,exist_ok=True)
    assets=args.assets_dir or args.workdir/'downloads';assets.mkdir(parents=True,exist_ok=True)
    tasks=['1','2'] if args.task=='both' else [args.task]
    names=['NGH-Seis_v1.0_IID_test_bundle.zip']
    for task in tasks:names.extend([f'task{task}_iid_deeplabv3plus.pt',f'task{task}_iid_deeplabv3plus_reference.json'])
    for name in names:
        info=config['assets'][name];path=assets/name
        if not valid(path,info):
            if args.assets_dir:p.error(f'Missing or corrupt local release asset: {name}')
            print(f'Downloading {name} ({info["size_bytes"]/1e6:.1f} MB)',flush=True)
            download(config['base_url']+name,path,info)
    release,audit=extract_bundle(assets/names[0],args.workdir/'data')
    (args.workdir/'integrity.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    if args.verify_only:
        print(json.dumps(audit,indent=2));return
    results={}
    for task in tasks:
        destination=args.workdir/f'task{task}_evaluation.json'
        subprocess.run([sys.executable,str(PROJECT/'scripts/evaluate_checkpoint.py'),
            '--release',str(release/'metadata'),'--task',task,'--device',args.device,
            '--checkpoint',str(assets/f'task{task}_iid_deeplabv3plus.pt'),
            '--expected',str(assets/f'task{task}_iid_deeplabv3plus_reference.json'),
            '--output',str(destination)],check=True)
        result=json.loads(destination.read_text(encoding='utf-8'))
        precision=config['reported_table_decimal_places']['task'+task]
        reference=config['reported_results']['task'+task]
        rounded={k:dict(decimal_places=digits,reported=f'{reference[k]:.{digits}f}',
                    reproduced=f'{result["metrics"][k]:.{digits}f}',
                    matches=f'{reference[k]:.{digits}f}'==f'{result["metrics"][k]:.{digits}f}')
                 for k,digits in precision.items()}
        results[task]=dict(device=result['device'],evaluation_amp=result['evaluation_amp'],
            checkpoint_sha256=result['checkpoint_sha256'],test_samples=result['test_sample_count'],
            metrics=result['metrics'],differences_from_report=result['differences_from_archived_test'],
            reported_table_comparison=rounded)
    report=dict(dataset_doi=config['dataset_doi'],verification_scope='Original-checkpoint evaluation on the complete fixed IID test split; training not repeated',
                integrity=audit,tasks=results)
    (args.workdir/'reported_result_reproduction.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Completed original-checkpoint evaluation; exact metric differences are recorded. CPU FP32 and CUDA AMP may differ.')
    if not all(v['matches'] for task in results.values() for v in task['reported_table_comparison'].values()):
        raise SystemExit('Reported table precision did not match; inspect the saved differences and environment.')

if __name__=='__main__':main()
