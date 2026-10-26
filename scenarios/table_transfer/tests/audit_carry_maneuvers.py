"""Independently audit new carry paths joined to saved adjacent task stages."""
from pathlib import Path
import tempfile,subprocess,json,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tests/artifacts/carry_maneuver_scan';OLD=ROOT/'tests/artifacts/random_room_resident'
for folder in sorted(OUT.glob('case_*')):
 if not (folder/'carry.npz').exists():continue
 old=OLD/folder.name/'transport';side=json.loads((OLD/folder.name/'replay/result.json').read_text())['selected_arm']
 with tempfile.TemporaryDirectory(dir=OUT) as directory:
  scratch=Path(directory)
  for name in ['result.json','lift.npz','place.npz','withdraw.npz']:
   if (old/name).exists():(scratch/name).symlink_to(old/name)
  (scratch/'carry.npz').symlink_to(folder/'carry.npz')
  result=subprocess.run([sys.executable,str(ROOT/'tests/audit_table_transport.py'),'--placement','--partial','--directory',str(scratch),'--arm',side],capture_output=True,text=True)
  if result.returncode:raise RuntimeError(folder.name+' '+result.stderr)
  audit=json.loads((scratch/'independent_audit.json').read_text());audit['full_task']=bool((old/'withdraw.npz').exists())
  (folder/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
  print(folder.name,'passed',flush=True)
