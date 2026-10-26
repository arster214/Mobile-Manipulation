#!/usr/bin/env python3
"""Convert the former MoveIt obstacle scene to Isaac Sim USD and cuRobo YAML."""

from pathlib import Path
from os.path import relpath

from isaacsim import SimulationApp

from rm75_obstacles import ObstacleScene, load_obstacle_scene


ROOT = Path(__file__).resolve().parents[1]
OBSTACLE_USD = ROOT / "assets" / "usd" / "obstacles" / "simple_obstacles.usda"
SCENE_USD = ROOT / "assets" / "usd" / "scenes" / "rm75_simple_obstacles.usda"
ROBOT_USD = ROOT / "assets" / "usd" / "rm75_dual_rmg24_sim" / "rm75_dual_rmg24_sim.usda"
CUROBO_YAML = ROOT / "config" / "simple_obstacles.yml"


def create_obstacle_usd(scene: ObstacleScene) -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    OBSTACLE_USD.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(OBSTACLE_USD))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Obstacles")
    stage.SetDefaultPrim(root.GetPrim())

    for item in scene.cuboids:
        cube = UsdGeom.Cube.Define(stage, f"/Obstacles/{item.name}")
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(*item.color_rgba[:3])])
        xform = UsdGeom.Xformable(cube)
        xform.AddTranslateOp().Set(Gf.Vec3d(*item.position))
        qw, qx, qy, qz = item.quaternion_wxyz
        xform.AddOrientOp().Set(Gf.Quatf(qw, qx, qy, qz))
        xform.AddScaleOp().Set(Gf.Vec3f(*item.dims))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        cube.GetPrim().SetCustomDataByKey("curobo:name", item.name)

    stage.GetRootLayer().Save()


def create_composed_scene() -> None:
    from pxr import Usd, UsdGeom

    SCENE_USD.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(SCENE_USD))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    robot = stage.DefinePrim("/World/Robot", "Xform")
    robot.GetReferences().AddReference(relpath(ROBOT_USD, SCENE_USD.parent))
    obstacles = stage.DefinePrim("/World/Obstacles", "Xform")
    obstacles.GetReferences().AddReference(relpath(OBSTACLE_USD, SCENE_USD.parent))
    stage.GetRootLayer().Save()


def main() -> None:
    scene = load_obstacle_scene(CUROBO_YAML)
    create_obstacle_usd(scene)
    create_composed_scene()
    print(f"Obstacle USD: {OBSTACLE_USD}")
    print(f"Composed scene USD: {SCENE_USD}")
    print(f"Canonical obstacle input: {CUROBO_YAML}")


simulation_app = SimulationApp({"headless": True})
try:
    main()
finally:
    simulation_app.close()
