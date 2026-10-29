"""One compact acceptance index. Never combine synthetic and physical pass rates."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tests/artifacts/pnp'
rows=[]
for name in ('final_suite','serial_object_offset','serial_target_offset','resume_physics','final_object_replay','final_target_replay'):
 path=OUT/(name+'.json')
 if not path.exists():continue
 for row in json.loads(path.read_text()):
  r=row['result'];rows.append({'suite':name,'case':row['case'],'mode':r['mode'],
    'status':r['status'],'reason':r['reason'],'holding':r['holding'],
    'first_attempt_success':r['first_attempt_success'],'recovered_success':r['recovered_success'],
    'first_failure':r['first_failure'],'elapsed_s':r['elapsed_s'],'retries':r['retries'],
    'physical_readback':row.get('physical_readback'),'config_sha256':r['config_sha256'],'log':r['log'],'run_id':r['run_id']})
model=json.loads((OUT/'contract/normal_cases.json').read_text())
source=hashlib.sha256()
for path in sorted([*ROOT.glob('pnp/*.py'),*ROOT.glob('planning/*.py'),ROOT/'pnp/config.json',ROOT/'planning/config.json']):
 source.update(str(path.relative_to(ROOT)).encode());source.update(path.read_bytes())
report={'scope':'Assisted object physics, not physical RM75 controller/frictional grip acceptance',
 'source_digest_at_report':source.hexdigest(),'distinct_normal_physical_configurations':3,
 'normal_scope':'baseline; source cube x=-1.03; placement slot x offset +0.02; all other scene/config values unchanged',
 'physical_note':'Replay/recovery/second-call rows are additional checks, not additional randomized normal configurations',
 'physics':rows,
 'synthetic_feedback':{'seed':29,'normal_cases':len(model),'successful':sum(r['result']['status']=='succeeded' for r in model),
  'evidence':'contract/normal_cases.json','claim':'interface and geometry only; not IK/robot success'},
 'numerical_baseline':json.loads((OUT/'numerical_audit/result.json').read_text()),
 'unfinished_cases':[name for name in ('serial_object_offset','serial_target_offset') if not (OUT/(name+'.json')).exists()]}
(OUT/'acceptance.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'physics_rows':len(rows),'synthetic_normal_cases':len(model),'unfinished':report['unfinished_cases']},ensure_ascii=False))
