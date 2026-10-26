"""Ten collision-free room starts, sequential isolated resident full-task trials."""
from pathlib import Path
import json, subprocess, sys
import numpy as np
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planning.interfaces import RobotState, EnvironmentState
from planning.curobo_backend import CuroboPlanner
from planning.model import state_vector
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'tests/artifacts/random_room_resident'

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 scene=json.loads((ROOT/'scene_config.json').read_text());settings=json.loads((ROOT/'planning/config.json').read_text())
 saved=json.loads((ROOT/'tests/artifacts/grasp_priority/result.json').read_text())['start_state']
 state=RobotState(saved['joint_names'],np.array(saved['positions']),np.zeros(len(saved['positions'])),0.,np.array(saved['base_pose']))
 world=yaml.safe_load((ROOT/'static_world.yml').read_text());world['cuboid']['Cup']={'dims':[.06,.06,.12],'pose':[-1.05,1.32,.81,1,0,0,0]}
 world['cuboid'].pop('Floor',None)
 planner=CuroboPlanner(Path(scene['robot']['curobo_config']),state,EnvironmentState(world,0.),settings,OUT/'sampling')
 rng=np.random.default_rng(20261027);samples=[]
 # Stratify into ten room cells so random samples cover different regions.
 for row in range(2):
  for col in range(5):
   for _ in range(1000):
    x=rng.uniform(-3.1+col*1.24,-3.1+(col+1)*1.24);y=rng.uniform(-3.1+row*3.1,-3.1+(row+1)*3.1);yaw=rng.uniform(-np.pi,np.pi)
    q=state_vector(state,planner.names);q[:3]=[x,y,yaw]
    if planner.collision_free(q[None]):samples.append([x,y,yaw]);break
   else:raise RuntimeError('No collision-free sample in cell')
 (OUT/'starts.json').write_text(json.dumps({'seed':20261027,'base_xy_yaw':samples,'scope':'Fixed upper-body posture; random collision-free base XY and yaw in stratified room cells'},indent=2)+'\n')
 # Release sampling process CUDA memory before spawning separate case workers.
 return samples

if __name__=='__main__':
 if '--sample' in sys.argv:main()
 else:
  subprocess.run([sys.executable,__file__,'--sample'],check=True)
  samples=json.loads((OUT/'starts.json').read_text())['base_xy_yaw']
  results=[]
  for i,base in enumerate(samples):
   folder=OUT/f'case_{i+1:02}';folder.mkdir(exist_ok=True)
   with (folder/'run.log').open('w') as log:
    process=subprocess.run([sys.executable,str(ROOT/'tests/benchmark_resident_planning.py'),'full','--initial-base',*map(str,base),'--output',str(folder),'--repeats','1'],stdout=log,stderr=subprocess.STDOUT)
   row={'case':i+1,'base_xy_yaw':base,'exit_code':process.returncode}
   if (folder/'timing.json').exists():row['timing']=json.loads((folder/'timing.json').read_text())
   results.append(row);(OUT/'summary.json').write_text(json.dumps(results,indent=2)+'\n');print('CASE',i+1,'EXIT',process.returncode,flush=True)
