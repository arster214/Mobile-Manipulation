#!/usr/bin/env python3
from pathlib import Path
import sys
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def generate_launch_description():
    pkg = Path(__file__).resolve().parents[1]
    config = pkg / 'config'
    robot_description = {'robot_description': (config / 'overseas_75_b_v_description.urdf').read_text()}
    robot_description_semantic = {'robot_description_semantic': (config / 'overseas_75_b_v_description.srdf').read_text()}
    robot_description_kinematics = {'robot_description_kinematics': load_yaml(config / 'kinematics.yaml')}
    joint_limits = {'robot_description_planning': load_yaml(config / 'joint_limits.yaml')}
    planning_pipelines = {
        'planning_pipelines': ['ompl'],
        'default_planning_pipeline': 'ompl',
        'ompl': load_yaml(config / 'ompl_planning.yaml'),
    }
    trajectory_execution = load_yaml(config / 'moveit_controllers.yaml')

    move_group_params = [
        robot_description,
        robot_description_semantic,
        robot_description_kinematics,
        joint_limits,
        planning_pipelines,
        trajectory_execution,
        {
            'publish_robot_description_semantic': True,
            'allow_trajectory_execution': LaunchConfiguration('allow_trajectory_execution'),
            'publish_planning_scene': True,
            'publish_geometry_updates': True,
            'publish_state_updates': True,
            'publish_transforms_updates': True,
            'monitor_dynamics': False,
        },
    ]

    return LaunchDescription([
        DeclareLaunchArgument('use_rsp', default_value='true'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('use_move_group', default_value='true'),
        DeclareLaunchArgument('use_obstacles', default_value='true'),
        DeclareLaunchArgument('move_group_namespace', default_value=''),
        DeclareLaunchArgument('disabled_capabilities', default_value=''),
        DeclareLaunchArgument('allow_trajectory_execution', default_value='false'),
        Node(package='tf2_ros', executable='static_transform_publisher',
             arguments=['0','0','0','0','0','0','world','base_link_underpan']),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             condition=IfCondition(LaunchConfiguration('use_rsp')),
             parameters=[robot_description, {'publish_frequency': 30.0}], output='screen'),
        Node(package='moveit_ros_move_group', executable='move_group',
             condition=IfCondition(LaunchConfiguration('use_move_group')),
             namespace=LaunchConfiguration('move_group_namespace'),
             remappings=[('joint_states', '/joint_states')],
             parameters=move_group_params + [{
                 'disable_capabilities': LaunchConfiguration('disabled_capabilities'),
             }], output='screen'),
        TimerAction(period=0.75, actions=[
            ExecuteProcess(
                condition=IfCondition(LaunchConfiguration('use_obstacles')),
                cmd=[sys.executable, str(pkg.parents[1] / 'scripts' / 'rm75_moveit_obstacles.py'),
                     '--ros-args', '-r', '__ns:=/curobo'],
                output='screen',
            ),
        ]),
        TimerAction(period=2.5, actions=[
            Node(package='rviz2', executable='rviz2',
                 condition=IfCondition(LaunchConfiguration('use_rviz')),
                 arguments=['-d', str(config / 'moveit.rviz')],
                 parameters=[robot_description, robot_description_semantic, robot_description_kinematics,
                             joint_limits, planning_pipelines, {'MoveIt_Allow_External_Program': True}],
                 output='screen'),
        ]),
    ])
