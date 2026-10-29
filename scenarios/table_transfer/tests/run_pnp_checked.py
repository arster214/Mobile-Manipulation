"""Serial test launcher with host RAM telemetry; terminate only its own process group."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--min-available-mb',type=int,default=2048);p.add_argument('--label',required=True);args,extra=p.parse_known_args()
root=Path(__file__).resolve().parents[1];out=root/'tests/artifacts/pnp';out.mkdir(parents=True,exist_ok=True)
(out/(args.label+'.json')).unlink(missing_ok=True)
(out/(args.label+'_aborted.json')).unlink(missing_ok=True)
command=['/home/fang/isaacsim/python.sh',str(root/'tests/test_pnp_physx.py'),'--label',args.label,*extra]
env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
def available_mb():
 values={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','MemTotal:'))}
 return values['MemAvailable']/1024
if available_mb()<args.min_available_mb:raise SystemExit('Insufficient host memory before startup')
with (out/(args.label+'.log')).open('w') as log,(out/(args.label+'_resources.jsonl')).open('w') as telemetry:
 child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 minimum=available_mb();reason=None;started=time.monotonic()
 try:
  while child.poll() is None:
   available=available_mb();minimum=min(minimum,available)
   rss=0
   for item in Path('/proc').iterdir():
    if not item.name.isdigit():continue
    try:
     if os.getpgid(int(item.name))==child.pid:
      for line in (item/'status').read_text().splitlines():
       if line.startswith('VmRSS:'):rss+=int(line.split()[1])
    except (ProcessLookupError,FileNotFoundError,PermissionError):pass
   telemetry.write(json.dumps({'elapsed_s':time.monotonic()-started,'host_available_mb':available,'test_group_rss_mb':rss/1024})+'\n');telemetry.flush()
   if available<args.min_available_mb:
    reason='host_memory_guard';break
   time.sleep(1)
 except KeyboardInterrupt:
  reason='cancelled_by_test_operator'
 finally:
  if child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 summary={'returncode':child.returncode,'reason':reason,'minimum_available_mb':minimum,'elapsed_s':time.monotonic()-started,'command':command}
 if reason is not None:
  aborted={'status':'cancelled' if reason=='cancelled_by_test_operator' else 'failed',
           'reason':reason,'holding':'unknown','final_stage':'external_resource_guard',
           'elapsed_s':summary['elapsed_s'],'run_id':args.label,'retries':None,
           'last_state':None,'log':str(out/(args.label+'.log'))}
  (out/(args.label+'_aborted.json')).write_text(json.dumps(aborted,indent=2))
 (out/(args.label+'_resources.json')).write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary),flush=True)
 raise SystemExit(child.returncode if reason is None else 75)
