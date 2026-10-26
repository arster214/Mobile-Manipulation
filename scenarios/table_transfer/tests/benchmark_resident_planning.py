"""Same-snapshot resident benchmark; first complete pass warms all used solvers."""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
from dataclasses import asdict, is_dataclass
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planning import curobo_backend as backend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['grasp', 'placement', 'full'])
    parser.add_argument('--initial-base', type=float, nargs=3)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if args.stage in ('grasp', 'full'):
        import test_grasp_priority as experiment
    else:
        import test_arm_placement as experiment
    output = args.output or Path(__file__).parent / 'artifacts/resident_benchmark' / args.stage
    experiment.OUT = output / 'replay'
    original = backend.CuroboPlanner
    signature = inspect.signature(original)
    cache = {}
    counters = {'model_init_s': 0., 'ik_init_s': 0., 'models_created': 0, 'ik_created': 0}
    original_ik = original.ik.fget

    def ik(instance):
        if instance._ik is not None:
            return instance._ik
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = original_ik(instance)
        torch.cuda.synchronize()
        counters['ik_init_s'] += time.perf_counter() - started
        counters['ik_created'] += 1
        return result

    def resident(*positional, **keywords):
        bound = signature.bind(*positional, **keywords)
        bound.apply_defaults()
        def encode(value):
            if is_dataclass(value):
                return asdict(value)
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, Path):
                return str(value)
            raise TypeError(type(value))
        key = hashlib.sha256(json.dumps(bound.arguments, default=encode, sort_keys=True).encode()).hexdigest()
        if key not in cache:
            torch.cuda.synchronize()
            started = time.perf_counter()
            cache[key] = original(*positional, **keywords)
            torch.cuda.synchronize()
            counters['model_init_s'] += time.perf_counter() - started
            counters['models_created'] += 1
        planner = cache[key]
        planner.rng = np.random.default_rng(0)
        planner.extra_validation = None
        planner.motion_phase = 'approach'
        planner.search_stats = {}
        return planner

    original.ik = property(ik)
    backend.CuroboPlanner = resident
    experiment.CuroboPlanner = resident
    if args.stage == 'full':
        from planning import placement
        import test_arm_placement as transport
        transport.CuroboPlanner = resident
        transport.OLD = experiment.OUT
        transport.OUT = output / 'transport'
    torch.cuda.init()
    trials = []
    baseline_arrays = None
    for index in range(args.repeats + 1):
        # Reset search seeds, retaining models, optimizers and GPU allocations.
        for planner in cache.values():
            planner.rng = np.random.default_rng(0)
        before = counters.copy()
        torch.cuda.synchronize()
        started = time.perf_counter()
        if args.stage in ('grasp', 'full'):
            experiment.main(initial_base=args.initial_base)
        else:
            experiment.main()
        grasp_elapsed = time.perf_counter() - started
        grasp_report = json.loads((experiment.OUT / 'result.json').read_text())
        transport_ran = args.stage == 'full' and grasp_report.get('selected_arm') is not None
        failure = None
        if transport_ran:
            try:
                transport.main()
            except RuntimeError as error:
                failure = str(error)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        report = json.loads((experiment.OUT / 'result.json').read_text())
        if transport_ran:
            report = json.loads((transport.OUT / 'result.json').read_text())
        arrays = {str(path.relative_to(output)): {k: value.copy() for k, value in np.load(path).items()}
                  for path in output.rglob('*.npz')}
        if baseline_arrays is None:
            baseline_arrays = arrays
        same = arrays.keys() == baseline_arrays.keys() and all(
            values.keys() == baseline_arrays[path].keys() and all(
                np.array_equal(value, baseline_arrays[path][key]) for key, value in values.items())
            for path, values in arrays.items())
        trial = {'phase': 'initialization_and_warmup' if index == 0 else 'resident',
                 'wall_s': elapsed, 'grasp_s': grasp_elapsed if args.stage != 'placement' else None, 'transport_s': elapsed - grasp_elapsed if transport_ran else None, 'initialization': {k: counters[k] - before[k] for k in counters},
                 'same_trajectory_as_warmup': same,
                 'failure': failure,
                 'success': report.get('complete', report.get('selected_arm') is not None),
                 'stage_timing': report.get('timing', {})}
        trials.append(trial)
        if index:
            assert trial['initialization']['models_created'] == 0, trial
            assert trial['initialization']['ik_created'] == 0, trial
            assert same, trial
        (output / 'timing.json').write_text(json.dumps({'stage': args.stage,
            'scope': 'Identical state, target and world replay; includes search, validation and test artifact I/O. Warmup excluded from resident trials; no cached trajectory results.',
            'trials': trials}, indent=2) + '\n')
        print('RESIDENT_TIMING', json.dumps(trial), flush=True)


if __name__ == '__main__':
    main()
