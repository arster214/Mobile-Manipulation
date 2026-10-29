"""Cancellable process boundary for CUDA work; no control commands in worker."""
from pathlib import Path
import json
import subprocess
import sys
from .contracts import Fault


class PlannerProcess:
    def __init__(self, directory, payload):
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True)
        (self.directory/'request.json').write_text(json.dumps(payload))
        self.log=(self.directory/'planner.log').open('w')
        self.process=subprocess.Popen([sys.executable,'-m','pnp.planner_worker',str(self.directory)],
            cwd=Path(__file__).resolve().parents[1],stdout=self.log,stderr=subprocess.STDOUT)

    def poll(self):
        code=self.process.poll()
        if code is None:return False
        self.log.close()
        output=self.directory/'result.json'
        if code or not output.exists():
            raise Fault('planner_error',f'Worker exit {code}; see {self.directory}/planner.log')
        result=json.loads(output.read_text())
        if not result['complete']:
            raise Fault(result['reason'],result['message'])
        return True

    def cancel(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=1.)
            except subprocess.TimeoutExpired:
                self.process.kill();self.process.wait()
        self.log.close()
