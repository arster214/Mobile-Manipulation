#!/usr/bin/env python3
# Alias kept for MoveIt habits: ros2 launch ... demo.launch.py
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def generate_launch_description():
    source = Path(__file__).with_name("moveit_frontend.launch.py")
    spec = spec_from_file_location("rm75_moveit_frontend_launch", source)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.generate_launch_description()
