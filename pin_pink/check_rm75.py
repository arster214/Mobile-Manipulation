"""Check RM75 geometry and local dual-arm IK with fixed and planar bases."""
from pathlib import Path

import numpy as np
import pinocchio as pin
import pink
from pink.tasks import FrameTask, PostureTask

WORKSPACE = Path(__file__).resolve().parents[1]
URDF = WORKSPACE / "realman_rm75/assets/urdf/rm75_dual_rmg24_curobo.urdf"
ARM_JOINTS = [f"{side}_joint{i}" for side in ("l", "r") for i in range(1, 8)]


def check(planar: bool) -> None:
    if planar:
        full = pin.buildModelFromUrdf(str(URDF), pin.JointModelPlanar())
    else:
        full = pin.buildModelFromUrdf(str(URDF))
    q = pin.neutral(full)
    q[full.joints[full.getJointId("platform_joint")].idx_q] = 0.5
    for name in full.names:
        if "finger" in name:
            q[full.joints[full.getJointId(name)].idx_q] = 0.0325
    keep = set(ARM_JOINTS + (["root_joint"] if planar else []))
    locks = [i for i in range(1, full.njoints) if full.names[i] not in keep]
    model = pin.buildReducedModel(full, locks, q)
    assert model.nv == (17 if planar else 14)
    q0 = pin.neutral(model)
    for side in ("l", "r"):
        for joint, value in ((2, 0.3), (4, -0.7)):
            q0[model.joints[model.getJointId(f"{side}_joint{joint}")].idx_q] = value
    config = pink.Configuration(model, model.createData(), q0)
    delta = np.zeros(model.nv)
    for name in ARM_JOINTS:
        delta[model.joints[model.getJointId(name)].idx_v] = 0.015
    if planar:
        delta[:3] = [0.01, 0.005, 0.01]
    goal = pink.Configuration(model, model.createData(), pin.integrate(model, q0, delta))
    tasks = []
    for frame in ("l_tool0", "r_tool0"):
        assert model.existFrame(frame)
        task = FrameTask(frame, position_cost=1.0, orientation_cost=1.0)
        task.set_target(goal.get_transform_frame_to_world(frame))
        tasks.append(task)
    posture = PostureTask(cost=1e-8)
    posture.set_target(q0)
    initial = max(float(np.linalg.norm(t.compute_error(config))) for t in tasks)
    for _ in range(300):
        v = pink.solve_ik(config, tasks + [posture], 0.01, solver="quadprog")
        assert np.all(np.isfinite(v))
        config.integrate_inplace(v, 0.01)
    final = max(float(np.linalg.norm(t.compute_error(config))) for t in tasks)
    assert final < 1e-4 and final < initial * 0.01, (initial, final)
    config.check_limits()
    print(f"{'Planar base' if planar else 'Fixed base'}: nq={model.nq}, nv={model.nv}, "
          f"dual-arm pose error {initial:.6e} -> {final:.6e}")


def main() -> None:
    print("URDF:", URDF)
    model = pin.buildModelFromUrdf(str(URDF))
    for geometry_type in (pin.GeometryType.VISUAL, pin.GeometryType.COLLISION):
        geometry = pin.buildGeomFromUrdf(
            model, str(URDF), geometry_type, package_dirs=[str(URDF.parent)]
        )
        assert geometry.ngeoms > 0
        print(f"{geometry_type}: {geometry.ngeoms} geometry objects loaded")
    check(planar=False)
    check(planar=True)
    print("PASS: RM75 assets, fixed-base and planar-base local dual-arm IK")
    print("Planar base is an SE(2) kinematic prototype; wheel constraints and collision avoidance are not configured.")


if __name__ == "__main__":
    main()
