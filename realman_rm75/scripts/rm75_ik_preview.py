"""Real-time translucent IK preview driven by the Robot Poser targets."""

from __future__ import annotations

PREVIEW_PATH = "/World/IKPreview"
TARGET_NAMES = {
    "left": "left_arm_poser",
    "right": "right_arm_poser",
}
CHAIN_LINKS = {
    "left": ("l_base_link1", "l_tool0"),
    "right": ("r_base_link1", "r_tool0"),
}


def _find_descendant(root, name: str):
    from pxr import Usd

    for prim in Usd.PrimRange(root):
        if prim.GetName() == name:
            return prim
    return None


def _robot_asset_reference(robot_prim) -> str:
    """Read the direct asset reference authored on /World/Robot."""
    for spec in robot_prim.GetPrimStack():
        references = list(spec.referenceList.prependedItems)
        if references and references[0].assetPath:
            return references[0].assetPath
    raise RuntimeError("/World/Robot has no authored asset reference")


def _set_world_transform(prim, world_transform) -> None:
    """Author a local matrix that places prim at the requested world pose."""
    from pxr import Gf, Usd, UsdGeom

    parent = prim.GetParent()
    parent_world = Gf.Matrix4d(1.0)
    if parent.IsValid():
        parent_world = UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
    local_transform = Gf.Matrix4d(world_transform) * parent_world.GetInverse()
    translation = local_transform.ExtractTranslation()
    rotation = local_transform.ExtractRotation().GetQuat()
    orientation = Gf.Quatd(rotation.GetReal(), rotation.GetImaginary())

    xformable = UsdGeom.Xformable(prim)
    double = UsdGeom.XformOp.PrecisionDouble
    float_ = UsdGeom.XformOp.PrecisionFloat
    translate_precision = double
    orient_precision = double
    scale_precision = double
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_precision = op.GetPrecision()
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            orient_precision = op.GetPrecision()
        elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_precision = op.GetPrecision()

    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp(translate_precision).Set(
        Gf.Vec3f(translation) if translate_precision == float_ else translation
    )
    xformable.AddOrientOp(orient_precision).Set(
        Gf.Quatf(orientation) if orient_precision == float_ else orientation
    )
    xformable.AddScaleOp(scale_precision).Set(
        Gf.Vec3f(1.0) if scale_precision == float_ else Gf.Vec3d(1.0)
    )


class RM75IKPreview:
    """Solve both arm targets with Isaac IK and display the result as a ghost."""

    def __init__(self, stage, robot_prim) -> None:
        from isaacsim.robot.poser import robot_poser
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

        self._stage = stage
        self._robot_prim = robot_prim
        self._last_targets = {}
        self._last_solve_success = {}
        self._posers = {}
        self._real_starts = {}

        stage.RemovePrim(PREVIEW_PATH)
        preview = UsdGeom.Xform.Define(stage, PREVIEW_PATH).GetPrim()
        preview.GetReferences().AddReference(_robot_asset_reference(robot_prim))
        preview.GetVariantSet("Physics").SetVariantSelection("physics")
        self._preview_prim = preview

        material = UsdShade.Material.Define(stage, f"{PREVIEW_PATH}/PreviewMaterial")
        shader = UsdShade.Shader.Define(stage, f"{PREVIEW_PATH}/PreviewMaterial/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.1, 0.85, 1.0)
        )
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.28)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        for prim in Usd.PrimRange(preview):
            if prim == preview:
                continue
            if prim.IsA(UsdGeom.Gprim):
                path = str(prim.GetPath())
                is_arm_visual = "/l_base_link1/" in path or "/r_base_link1/" in path
                if not is_arm_visual:
                    UsdGeom.Imageable(prim).MakeInvisible()
                    continue
                binding = UsdShade.MaterialBindingAPI(prim)
                binding.UnbindAllBindings()
                binding.Bind(material, UsdShade.Tokens.strongerThanDescendants)
                gprim = UsdGeom.Gprim(prim)
                gprim.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.85, 1.0)])
                gprim.CreateDisplayOpacityAttr([0.28])
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)

        # The preview is not simulated, so copy the complete measured robot
        # pose before building either local IK chain. Copying every body keeps
        # the lift, shoulders, and arm links in one coherent kinematic state.
        for real_body in Usd.PrimRange(robot_prim):
            if not real_body.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            relative_path = str(real_body.GetPath())[len(str(robot_prim.GetPath())) :]
            preview_body = stage.GetPrimAtPath(PREVIEW_PATH + relative_path)
            if not preview_body.IsValid():
                continue
            real_world = UsdGeom.Xformable(real_body).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            _set_world_transform(preview_body, real_world)

        for side, (start_name, end_name) in CHAIN_LINKS.items():
            start = _find_descendant(preview, start_name)
            end = _find_descendant(preview, end_name)
            real_start = _find_descendant(robot_prim, start_name)
            if start is None or end is None:
                raise RuntimeError(f"IK preview chain is missing: {start_name} -> {end_name}")
            if real_start is None:
                raise RuntimeError(f"Real robot chain start is missing: {start_name}")

            # The preview articulation is excluded from PhysX, so its lift
            # joint remains at the asset default. Anchor each IK chain at the
            # measured flange-base pose used by the real robot instead.
            poser = robot_poser.RobotPoser(stage, preview, start, end)
            if not poser.joints:
                raise RuntimeError(f"IK preview chain has no joints: {side}")
            source_pose = robot_poser.get_named_pose(stage, robot_prim, TARGET_NAMES[side])
            if source_pose is not None and source_pose.joints:
                seed_by_name = {
                    Sdf.Path(path).name: value for path, value in source_pose.joints.items()
                }
                poser.set_seed(
                    [seed_by_name.get(Sdf.Path(joint.prim_path).name, 0.0) for joint in poser.joints]
                )
            self._posers[side] = poser
            self._real_starts[side] = real_start

        # Keep the joint definitions readable by Isaac FK while excluding
        # them from PhysX simulation.
        physics_scope = stage.GetPrimAtPath(f"{PREVIEW_PATH}/Physics")
        if physics_scope.IsValid():
            for prim in Usd.PrimRange(physics_scope):
                if prim.IsA(UsdPhysics.Joint):
                    UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)

        print("Installed real-time IK preview: /World/IKPreview")

    @staticmethod
    def _target_key(pose) -> tuple[float, ...] | None:
        if pose.target_position is None or pose.target_orientation is None:
            return None
        return tuple(float(value) for value in (*pose.target_position, *pose.target_orientation))

    def update(self) -> None:
        """Re-solve only targets whose transform changed since the last frame."""
        from isaacsim.robot.poser import robot_poser
        from pxr import Gf, Usd, UsdGeom
        from usd.schema.isaac.robot_schema import math as poser_math

        for side, pose_name in TARGET_NAMES.items():
            real_start_world = UsdGeom.Xformable(
                self._real_starts[side]
            ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            preview_start = self._posers[side].start_prim
            preview_start_world = UsdGeom.Xformable(
                preview_start
            ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            real_position = Gf.Matrix4d(real_start_world).ExtractTranslation()
            preview_position = Gf.Matrix4d(preview_start_world).ExtractTranslation()
            base_moved = (real_position - preview_position).GetLength() > 1e-6
            if base_moved:
                _set_world_transform(preview_start, real_start_world)
                self._last_targets.pop(side, None)

            pose = robot_poser.get_named_pose(self._stage, self._robot_prim, pose_name)
            if pose is None or not pose.success:
                continue
            target_key = self._target_key(pose)
            if target_key is None or target_key == self._last_targets.get(side):
                continue

            target = poser_math.Transform(
                t=pose.target_position,
                q=pose.target_orientation,
            )
            result = self._posers[side].solve_ik(target)
            self._last_solve_success[side] = bool(result.success)
            if result.success:
                self._posers[side].chain.teleport(result.joints)
            self._last_targets[side] = target_key

    def last_solve_succeeded(self, side: str) -> bool | None:
        """Return the latest IK outcome, or None before that side is solved."""
        if side not in TARGET_NAMES:
            raise ValueError(f"Unknown IK preview side: {side}")
        return self._last_solve_success.get(side)

    def destroy(self) -> None:
        self._posers.clear()
        self._real_starts.clear()
        self._last_solve_success.clear()
        if self._stage is not None:
            self._stage.RemovePrim(PREVIEW_PATH)
        self._stage = None
        self._preview_prim = None
