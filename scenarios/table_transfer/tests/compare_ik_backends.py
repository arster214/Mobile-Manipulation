"""Equal task/seed/acceptance-budget comparison; no executed robot motion."""
import json,time,subprocess,collections,os,hashlib
from pathlib import Path
import numpy as np
import pinocchio as pin
import pink
from pink.tasks import FrameTask,PostureTask
from pink.barriers import Barrier
from ik_compare_common import Problem,ROOT,SOURCE,OUT

class SphereBarrier(Barrier):
 def __init__(self,p):
  super().__init__(dim=96,gain=20.);self.p=p
 def compute_barrier(self,configuration):
  raise NotImplementedError("Uses direct QP inequality assembly")
 def compute_jacobian(self,configuration):
  raise NotImplementedError("Uses direct QP inequality assembly")
 def compute_qp_objective(self,configuration):
  return np.zeros((self.p.model.nv,self.p.model.nv)),np.zeros(self.p.model.nv)
 def compute_qp_inequalities(self,configuration,dt):
  p=self.p;c,J,sg,wg,delta,dist,d,pos,norm=p.geometry(configuration.q,True)
  si=np.argsort(sg)[:64];a,b=p.pairs[si].T
  normals=delta[si]/np.maximum(dist[si,None],1e-12)
  sj=np.einsum('ni,nij->nj',normals,J[a]-J[b])
  flat=np.argsort(wg,axis=None)[:32];i,j=np.unravel_index(flat,wg.shape)
  normal=pos[i,j]/np.maximum(norm[i,j,None],1e-12)
  inside=norm[i,j]<1e-12
  if np.any(inside):normal[inside]=np.eye(3)[np.argmax(d[i[inside],j[inside]],axis=1)]
  normal*=np.sign(c[i]-p.box_c[j]);wj=np.einsum('ni,nij->nj',normal,J[i])
  gap=np.concatenate([sg[si],wg[i,j]])
  return -np.concatenate([sj,wj])/dt,20.*gap

p=Problem();banks=p.banks();np.save(OUT/'shared_seed_banks.npy',banks)
barrier=SphereBarrier(p)
# Validate sphere-center Jacobian against finite differences, independent of QP.
q=np.array(p.reference['rows'][0]['start_q']);c,J,*_=p.geometry(q,True);maxerr=0.
for j in range(p.model.nv):
 qp=q.copy();qm=q.copy();qp[j]+=1e-6;qm[j]-=1e-6
 cp=p.geometry(qp)[0];cm=p.geometry(qm)[0]
 maxerr=max(maxerr,float(np.max(np.abs((cp-cm)/2e-6-J[:,:,j]))))
assert maxerr<1e-6,maxerr
print('JACOBIAN_VERIFIED',maxerr,flush=True)
BUDGET=1.;MAX_ITER=100;REPEATS=3
results=[]
# Pink warm-up, excluded from measurements.
warm=pink.Configuration(p.model,p.model.createData(),q.copy());task=FrameTask('r_grasp_tcp',position_cost=1.,orientation_cost=1.,lm_damping=1e-3);task.set_target(pin.SE3(p.target[:3,:3],p.target[:3,3]));rest=PostureTask(cost=1e-6);rest.set_target(q)
try:pink.solve_ik(warm,[task,rest],.05,solver='quadprog',barriers=[barrier])
except Exception:pass
for repeat in range(REPEATS):
 for index,row in enumerate(p.reference['rows']):
  started=time.perf_counter();accepted=False;tried=0;steps=0;failures=collections.Counter();best=None;bestq=None
  for seed in banks[index].reshape(-1,11):
   if time.perf_counter()-started>=BUDGET:break
   tried+=1;config=pink.Configuration(p.model,p.model.createData(),seed.copy());rest.set_target(np.array(row['start_q']))
   last_error=float('inf');stagnant=0
   for iteration in range(MAX_ITER):
    if time.perf_counter()-started>=BUDGET:break
    try:v=pink.solve_ik(config,[task,rest],.05,solver='quadprog',barriers=[barrier]);config.integrate_inplace(v,.05)
    except Exception as exc:failures[type(exc).__name__]+=1;break
    steps+=1
    actual=config.get_transform_frame_to_world('r_grasp_tcp');pe=np.linalg.norm(actual.translation-p.target[:3,3]);re=np.linalg.norm(pin.log3(p.target[:3,:3].T@actual.rotation))
    if pe<=1e-4 and re<=1e-3:
     check=p.check(config.q)
     best=check;bestq=config.q.tolist()
     if check['accepted']:accepted=True
     else:failures['terminal_rejected']+=1
     break
    error=pe+re
    stagnant=stagnant+1 if abs(error-last_error)<1e-6 else 0;last_error=error
    if stagnant>=10:failures['stagnated']+=1;break
   if accepted:break
  elapsed=(time.perf_counter()-started)*1000
  result={'backend':'pink','repeat':repeat,'index':index+1,'accepted':accepted,'wall_ms':elapsed,'seeds_tried':tried,'iterations':steps,'failures':dict(failures),'end_q':bestq,'audit':best,'stop':'accepted' if accepted else ('time_budget' if elapsed>=1000 else 'seed_budget')}
  results.append(result);print(json.dumps({k:result[k] for k in ('backend','repeat','index','accepted','wall_ms','seeds_tried')}),flush=True)
 (OUT/'partial_pink.json').write_text(json.dumps(results,indent=2))
# Run the GPU comparison separately so GPU work does not contend with Pink timing.
worker=subprocess.Popen(['/home/fang/isaacsim/python.sh',str(ROOT/'tests/ik_compare_gpu_worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=(OUT/'gpu_worker_stderr.log').open('w'),text=True,bufsize=1)
def read(prefix):
 while True:
  line=worker.stdout.readline()
  if not line:raise RuntimeError('GPU worker ended unexpectedly; see stderr log')
  if line.startswith('COMPARE_ERROR'):raise RuntimeError(line)
  if line.startswith(prefix):return json.loads(line[len(prefix):])
ready=read('COMPARE_READY ')
try:
 for repeat in range(REPEATS):
  for index,row in enumerate(p.reference['rows']):
   started=time.perf_counter();accepted=False;best=None;bestq=None;gpu_ms=0.;tried=0
   for attempt in range(5):
    if time.perf_counter()-started>=BUDGET:break
    worker.stdin.write(json.dumps({'start':row['start_q'],'seeds':banks[index,attempt].tolist()})+'\n');worker.stdin.flush()
    answer=read('COMPARE_RESULT ');gpu_ms+=answer['solve_ms'];tried+=32
    for qout,success in zip(answer['solutions'],answer['solver_success']):
     # Apply exactly the same independent acceptance, even if GPU's metric differs.
     check=p.check(np.array(qout))
     if check['accepted']:
      accepted=True;best=check;bestq=qout;break
    if accepted:break
   elapsed=(time.perf_counter()-started)*1000
   result={'backend':'curobo','repeat':repeat,'index':index+1,'accepted':accepted,'wall_ms':elapsed,'gpu_solve_ms':gpu_ms,'seeds_tried':tried,'attempts':attempt+1,'end_q':bestq,'audit':best,'stop':'accepted' if accepted else ('time_budget' if elapsed>=1000 else 'seed_budget')}
   results.append(result);print(json.dumps({k:result[k] for k in ('backend','repeat','index','accepted','wall_ms','seeds_tried')}),flush=True)
finally:
 if worker.poll() is None:worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.wait(timeout=30)
report={'task':'same 20 starts and fixed target; three repetitions; 11 DoF; terminal poses only','seed_bank_sha256':hashlib.sha256((OUT/'shared_seed_banks.npy').read_bytes()).hexdigest(),'source_sha256':p.reference['source_sha256'],'sphere_count':len(p.radii),'self_pairs':len(p.pairs),'position_tol_m':1e-4,'rotation_tol_rad':1e-3,'heading_half_angle_deg':30,'max_seeds':160,'max_rounds':5,'per_query_budget_s':BUDGET,'budget_policy':'stop launching new steps or GPU rounds after budget; in-flight work may overrun','pink_max_iterations_per_seed':MAX_ITER,'pink_collision_constraint':'QP distance barriers on 64 nearest self pairs and 32 nearest world pairs per step; all pairs checked at acceptance','heading_constraint':'same hard final filter for both','gpu':ready,'jacobian_max_abs_error':maxerr,'timing':'warm wall time includes seed handling, solution checks and retries; curobo includes local worker IPC, also reported GPU synchronized solve time; setup and warmup excluded','rows':results}
(OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
