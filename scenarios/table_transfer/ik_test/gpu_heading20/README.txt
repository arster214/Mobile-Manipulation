Heading-constrained GPU IK candidate selection
Viewer http://127.0.0.1:8093/
Hard final acceptance: object bearing from final base center within +/-30 degrees of base local -Y forward.
Implementation uses returned candidate filtering and front-facing retry seeds, not an internal CUDA nonlinear inequality constraint.
Up to 5 calls per original start, each with 32 parallel seeds and internal LM/particle/L-BFGS iterations. Changed seeds on retries; original start remains current_state.
19/20 accepted after independent FK, joint limits, self/world collision and heading checks. Pair 14 rejected for approximately 46 degrees even though pose IK converged.
Timing includes synchronization, candidate transfer/filtering and retries; excludes setup/warmup and independent audit.
No nonholonomic or swept path validation. No hard positive clearance margin.
