"""Validate Pinocchio dynamics and Pink IK without external robot assets."""
from importlib.metadata import version

import numpy as np
import pinocchio as pin
import pink
import qpsolvers
from pink.tasks import FrameTask, PostureTask


def main():
    for package in ("pin", "pin-pink", "numpy", "qpsolvers", "quadprog", "meshcat"):
        print(f"{package}: {version(package)}")
    print("QP solvers:", qpsolvers.available_solvers)
    assert "quadprog" in qpsolvers.available_solvers

    model = pin.buildSampleModelManipulator()
    data = model.createData()
    q0 = pin.neutral(model)
    q0[1], q0[3] = 0.4, -0.8
    config = pink.Configuration(model, data, q0)
    frame = "effector_body"
    task = FrameTask(frame, position_cost=1.0, orientation_cost=1.0)
    task.set_target_from_configuration(config)
    # A reachable target pose generated from a nearby joint configuration.
    q_goal = q0 + np.array([0.05, -0.05, 0.03, 0.06, -0.03, 0.02])
    goal = pink.Configuration(model, model.createData(), q_goal)
    task.set_target(goal.get_transform_frame_to_world(frame))
    posture = PostureTask(cost=1e-6)
    posture.set_target(q0)
    initial = float(np.linalg.norm(task.compute_error(config)))
    dt = 0.01
    for _ in range(300):
        velocity = pink.solve_ik(config, [task, posture], dt, solver="quadprog")
        assert np.all(np.isfinite(velocity))
        config.integrate_inplace(velocity, dt)
    final = float(np.linalg.norm(task.compute_error(config)))
    assert final < 1e-4 and final < initial * 0.01, (initial, final)

    jacobian = pin.computeFrameJacobian(model, data, config.q, model.getFrameId(frame))
    mass = pin.crba(model, data, config.q)
    tau = pin.rnea(model, data, config.q, np.zeros(model.nv), np.zeros(model.nv))
    assert jacobian.shape == (6, model.nv)
    assert np.all(np.isfinite(tau))
    assert np.linalg.eigvalsh(mass).min() > 0
    print(f"6-DOF manipulator IK error: {initial:.6e} -> {final:.6e}")
    print("PASS: forward kinematics, Jacobian, mass matrix, inverse dynamics, Pink IK")


if __name__ == "__main__":
    main()
