"""Show translation and rotation handles together for RM75 pose targets."""

from __future__ import annotations

from collections.abc import Callable


_POSE_TARGET_SUFFIXES = (
    "/NamedPoses/left_arm_poser",
    "/NamedPoses/right_arm_poser",
)


class CombinedPoseGizmo:
    """Apply a scoped patch to Isaac's standard transform manipulator."""

    def __init__(self) -> None:
        import omni.usd
        from omni.kit.manipulator.transform import TransformManipulator

        self._manipulator_type = TransformManipulator
        self._original_update: Callable = TransformManipulator._update_from_model
        original_update = self._original_update

        def _update_with_combined_pose_handles(manipulator) -> None:
            original_update(manipulator)
            selected_paths = omni.usd.get_context().get_selection().get_selected_prim_paths()
            if not selected_paths or not selected_paths[0].endswith(_POSE_TARGET_SUFFIXES):
                return
            if manipulator._translate_gizmo is None or manipulator._rotation_gizmo is None:
                return

            manipulator._translate_gizmo.visible = True
            manipulator._rotation_gizmo.visible = True
            if manipulator._scale_gizmo is not None:
                manipulator._scale_gizmo.visible = False

        TransformManipulator._update_from_model = _update_with_combined_pose_handles

    def refresh(self) -> None:
        """Nudge the active operation setting after pose-target selection."""
        import carb.settings

        settings = carb.settings.get_settings()
        operation_path = "/app/transform/operation"
        operation = settings.get(operation_path) or "move"
        settings.set(operation_path, operation)

    def destroy(self) -> None:
        """Restore Isaac's unmodified manipulator behavior."""
        if self._manipulator_type is None:
            return
        self._manipulator_type._update_from_model = self._original_update
        self._manipulator_type = None


def install_combined_pose_gizmo() -> CombinedPoseGizmo:
    """Install the combined 6-DoF gizmo used by the two Robot Poser targets."""
    return CombinedPoseGizmo()
