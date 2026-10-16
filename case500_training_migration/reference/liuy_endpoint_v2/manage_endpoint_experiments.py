#!/usr/bin/env python3
"""Prepare and supervise 70 endpoint experiments, three jobs per physical GPU."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / 'data/case118_endpoint_training_v2'
SOURCE = REPO / 'data/case118_endpoint_id_1280k_v1_20261012'
RESULTS = REPO.parent / 'Results/0-1labels'
CONTROL = RESULTS / '_endpoint_v2_control'
SCALES = (10,20,40,45,50,55,60,80,90,100,110,120,160,320)
SEEDS = (11,22,33,44,55)
GPUS = (0,1,2)
SLOTS = tuple((gpu, slot) for gpu in GPUS for slot in range(3))
EXPERIMENT = 'case118_endpoint_v2_20261013'

def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20), b''):
            h.update(block)
    return h.hexdigest()

def canonical_hash(config):
    # Must match train_endpoint_run.py's public config identity function.
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def configure():
    """Only manifests/configuration are written here; training is never started."""
    CONTROL.mkdir(parents=True, exist_ok=True)
    parent_root = DATA / 'train'
    train_info = {}
    for size in SCALES:
        label = f'{size}k'
        index = parent_root / label / 'parent_ids.npy'
        graph_index = parent_root / label / 'source_graph_indices.npy'
        if not index.is_file() or not graph_index.is_file():
            raise FileNotFoundError(f'Missing frozen indices for {label}; run prepare first.')
        train_info[label] = {'parents_sha256':digest(index), 'graph_indices_sha256':digest(graph_index)}
    source_index = SOURCE / 'processed/canos_rho_scales/index.json'
    common = {'source_index_sha256':digest(source_index)}
    for split in ('val','test'):
        common[split + '_parents_sha256'] = digest(SOURCE / f'splits/{split}_parent_ids.npy')
    jobs = []
    for size in SCALES:
        label = f'{size}k'
        ident = {**common, **train_info[label], 'experiment_id':EXPERIMENT}
        data_hash = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()
        for seed in SEEDS:
            run_dir = RESULTS / label / f'seed{seed}'
            for folder in ('Model','Evaluation','Logs'):
                (run_dir / folder).mkdir(parents=True, exist_ok=True)
            config = {
                'repo_root':str(REPO), 'train_cache':str(DATA / f'processed/{label}/train.pt'),
                'val_cache':str(DATA / 'shared/val.pt'), 'test_cache':str(DATA / 'shared/test.pt'),
                'run_dir':str(run_dir), 'scale':label, 'seed':seed, 'epochs':300,
                'experiment_id':EXPERIMENT, 'data_hash':data_hash,
                'expected_train_count':size*1000, 'expected_val_count':16800,
                'expected_test_count':16800, 'device':'cuda:0', 'batch_size':64,
                'val_batch_size':50, 'val_every':2, 'lr':0.0005,
                'hidden_dim':128, 'k_steps':20, 'num_workers':0,
                'torch_threads':4,
                'train_parent_ids':str(DATA / f'train/{label}/parent_ids.npy'),
                'val_parent_ids':str(DATA / 'shared/val_parent_ids.npy'),
                'test_parent_ids':str(DATA / 'shared/test_parent_ids.npy'),
                'parent_manifest':str(SOURCE / 'manifests/parents.jsonl'),
            }
            config_path = CONTROL / 'configs' / f'{label}_seed{seed}.json'
            if config_path.exists() and read(config_path) != config:
                if (run_dir / 'Model/last.pt').exists() or (run_dir / 'Model/best.pt').exists():
                    raise RuntimeError(f'Config changed for existing run: {run_dir}')
            write(config_path, config)
            jobs.append({'id':f'{label}_seed{seed}','scale':label,'seed':seed,
                         'config':str(config_path),'run_dir':str(run_dir),'data_hash':data_hash})
    manifest = {'experiment_id':EXPERIMENT, 'gpu_ids':list(GPUS), 'jobs_per_gpu':3,
                'total_jobs':len(jobs),'scales_k':list(SCALES),'seeds':list(SEEDS),
                'holdout_records':{'val':16800,'test':16800},'jobs':jobs}
    write(CONTROL / 'manifest.json', manifest)
    print(f'Configured {len(jobs)} runs: {len(SCALES)} scales x {len(SEEDS)} seeds.',flush=True)
    return manifest

def completion(job):
    path = Path(job['run_dir']) / 'Evaluation/completion.json'
    if not path.exists(): return None
    result = read(path)
    if not result.get('finished') or result.get('smoke_only'): return None
    if result.get('data_hash') != job['data_hash']:
        raise RuntimeError(f'Completion data hash mismatch: {path}')
    if result.get('config_hash') != canonical_hash(read(job['config'])):
        raise RuntimeError(f'Completion config hash mismatch: {path}')
    # The trainer validates its full canonical config on resume; the manifest
    # additionally ties each completed run to the same frozen data and outputs.
    for name in ('Model/best.pt','Model/last.pt','Evaluation/test_metrics.json'):
        if not (Path(job['run_dir']) / name).is_file():
            raise RuntimeError(f'Completed run is missing {name}: {job["id"]}')
    return result

def summarize(manifest):
    rows = []
    for job in manifest['jobs']:
        comp = completion(job)
        progress_path = Path(job['run_dir']) / 'progress.json'
        progress = read(progress_path) if progress_path.exists() else {}
        row = {'run_id':job['id'],'scale':job['scale'],'seed':job['seed'],
               'status':'complete' if comp else progress.get('status','pending'),
               'epoch':progress.get('epoch',''),'best_val_mse':(comp or {}).get('best_val_mse',''),
               'best_epoch':(comp or {}).get('best_epoch',''),
               'training_seconds':(comp or {}).get('training_seconds','')}
        metric_path = Path(job['run_dir']) / 'Evaluation/test_metrics.json'
        if comp and metric_path.exists():
            metrics = read(metric_path)
            for key,value in metrics.get('overall',{}).items():
                if isinstance(value,(int,float)): row['test_'+key] = value
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    target = CONTROL / 'run_summary.csv'
    tmp = target.with_suffix('.csv.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    os.replace(tmp,target)
    # Mean/std across completed seeds, with n_seeds always visible.
    import statistics
    aggregate=[]
    for size in SCALES:
        group=[r for r in rows if r['scale']==f'{size}k' and r['status']=='complete']
        rec={'scale':f'{size}k','n_seeds':len(group)}
        for key in fields:
            if key.startswith('test_') or key=='best_val_mse':
                values=[float(r[key]) for r in group if isinstance(r.get(key),(float,int))]
                if values:
                    rec[key+'_mean']=statistics.mean(values)
                    rec[key+'_std']=statistics.stdev(values) if len(values)>1 else None
        aggregate.append(rec)
    write(CONTROL/'seed_summary.json',aggregate)
    return rows

def preflight(manifest):
    for job in manifest['jobs']:
        config = read(job['config'])
        for key in ('train_cache','val_cache','test_cache'):
            if not Path(config[key]).is_file(): raise FileNotFoundError(config[key])
    query = subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used',
                                     '--format=csv,noheader,nounits'], text=True)
    print(query, flush=True)
    lines=[line.split(',') for line in query.strip().splitlines()]
    if not set(GPUS).issubset({int(x[0]) for x in lines}): raise RuntimeError('GPU 0,1,2 required.')
    running = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
                                       '--format=csv,noheader'],text=True).strip()
    if running:
        raise RuntimeError('GPU compute processes already running; no new training launched:\n'+running)
    mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    if mem['MemAvailable'] < 110 * 1024**2: raise RuntimeError('Less than 110 GiB available RAM for 9 jobs.')
    free = os.statvfs(RESULTS)
    if free.f_bavail * free.f_frsize < 25 * 1024**3: raise RuntimeError('Less than 25 GiB disk space available.')

def run(prepare=False):
    CONTROL.mkdir(parents=True,exist_ok=True)
    lock=(CONTROL/'supervisor.lock').open('a+')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('The endpoint supervisor is already running.')
    stopping=False
    active={}
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
        print(f'{now()} Stop requested; no new runs will start.',flush=True)
        for entry in active.values():
            if entry['process'].poll() is None: entry['process'].send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    write(CONTROL/'supervisor.json',{'pid':os.getpid(),'started':now(),'status':'preparing' if prepare else 'starting'})
    if prepare:
        command=[sys.executable,str(HERE/'prepare_endpoint_data.py'),'--repo',str(REPO),'--build-caches']
        subprocess.run(command,cwd=REPO,check=True)
    if stopping: return 130
    manifest=configure()
    pending=[job for job in manifest['jobs'] if completion(job) is None]
    if not pending:
        summarize(manifest)
        write(CONTROL/'supervisor.json',{'pid':os.getpid(),'updated':now(),'status':'complete','active':[], 'pending':0})
        print('All 70 runs already complete.',flush=True)
        return 0
    preflight(manifest)
    print(f'{now()} Starting {len(pending)} pending runs. GPU 0/1/2: exactly 3 slots each.',flush=True)
    failed=[];finished=[]
    try:
        while pending or active:
            for gpu,slot in SLOTS:
                key=(gpu,slot)
                if stopping or not pending or key in active: continue
                job=pending.pop(0)
                run_dir=Path(job['run_dir']); log_path=run_dir/'Logs/console.log'
                log=log_path.open('a',buffering=1)
                log.write(f'\n{now()} GPU {gpu} SLOT {slot} {job["id"]}\n')
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',
                    MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',PYTHONUNBUFFERED='1')
                command=[sys.executable,'-u',str(HERE/'train_endpoint_run.py'),'--config',job['config']]
                proc=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
                active[key]={'process':proc,'job':job,'log':log,'started':now()}
                print(f'START gpu={gpu} slot={slot} pid={proc.pid} {job["id"]}',flush=True)
                # Stagger dataset loading; physical GPU assignment never changes.
                time.sleep(2)
            for key,entry in list(active.items()):
                code=entry['process'].poll()
                if code is None: continue
                job=entry['job'];entry['log'].close();del active[key]
                valid = code==0 and completion(job) is not None
                if valid: finished.append(job['id'])
                else: failed.append({'job':job['id'],'exit_code':code,'log':str(Path(job['run_dir'])/'Logs/console.log')})
                print(f'{"DONE" if valid else "STOPPED/FAILED"} {job["id"]} exit={code}',flush=True)
                summarize(manifest)
            state={'pid':os.getpid(),'updated':now(),'status':'stopping' if stopping else 'running',
                   'active':[{'gpu':g,'slot':s,'pid':e['process'].pid,'job':e['job']['id'],'started':e['started']}
                             for (g,s),e in active.items()],
                   'pending':len(pending),'finished_this_session':finished,'failed_this_session':failed}
            write(CONTROL/'supervisor.json',state)
            if stopping and not active: break
            if active: time.sleep(5)
        state['status']='stopped' if stopping else ('finished_with_failures' if failed else 'complete')
        state['active']=[];write(CONTROL/'supervisor.json',state)
        summarize(manifest)
        return 130 if stopping else (1 if failed else 0)
    finally:
        for entry in active.values():
            if entry['process'].poll() is None: entry['process'].send_signal(signal.SIGTERM)
        for entry in active.values():
            entry['process'].wait();entry['log'].close()
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

def status():
    path=CONTROL/'supervisor.json'
    if path.exists():
        state=read(path);print(json.dumps(state,indent=2,ensure_ascii=False))
        for item in state.get('active',[]):
            p=RESULTS/item['job'].rsplit('_seed',1)[0]/('seed'+item['job'].rsplit('_seed',1)[1])/'progress.json'
            if p.exists(): print(item['job'],json.dumps(read(p),ensure_ascii=False))
    else: print('Supervisor has not been started.')
    path=CONTROL/'manifest.json'
    if path.exists():
        manifest=read(path)
        done=sum(completion(job) is not None for job in manifest['jobs'])
        print(f'Completed: {done}/{manifest["total_jobs"]}; 640k/1280k/rho=0.5 are not queued.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('configure','run','status','stop','plan'))
    p.add_argument('--prepare',action='store_true')
    args=p.parse_args()
    if args.command=='configure': configure()
    elif args.command=='run': return run(args.prepare)
    elif args.command=='status': status()
    elif args.command=='stop':
        state=read(CONTROL/'supervisor.json');pid=int(state['pid'])
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
        if str(Path(__file__).resolve()).encode() not in cmd:
            raise RuntimeError('Saved PID is not this supervisor; no signal sent.')
        os.kill(pid,signal.SIGTERM);print(f'Graceful stop requested for supervisor {pid}.')
    else:
        print(json.dumps({'scales_k':SCALES,'seeds':SEEDS,'total_runs':70,
                          'slots':[{'gpu':g,'slot':s} for g,s in SLOTS],
                          'holdout_records':16800,'epochs':300},indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
