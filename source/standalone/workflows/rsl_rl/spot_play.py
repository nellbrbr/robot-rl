# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

"""Script to play a checkpoint if an RL agent from RSL-RL."""
from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os
import traceback
from datetime import datetime

import carb
import gymnasium as gym
import torch
#from omni.isaac.lab.envs import RLTaskEnvCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.string import resolve_matching_names_values
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from rsl_rl.runners import OnPolicyRunner

# import orbit.spot  # noqa: F401
# from omni.isaac.lab.envs.mdp.commands.commands_cfg import Se2GamepadCommandCfg, Se2KeyboardCommandCfg
from omni.isaac.lab.devices.keyboard.se2_keyboard import Se2Keyboard

def run_sim():
    """Play with RSL-RL agent."""

    # parse configuration
    # env_cfg: RLTaskEnvCfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)

    # query policy input option
    while True:
        policy_load_type = input(
            "Select a input device option: \n"
            + "(1) to use keyboard, \n"
            + "(2) to use game controller,\n"
            + "(3) to use default command manager (randomly sampled inputs)\n"
        )
        match policy_load_type:
            case "1":
                keyboard = Se2Keyboard()
                # KEYBOARD_CFG = Se2KeyboardCommandCfg(
                #     asset=SceneEntityCfg("robot", body_names="body"),
                #     vel_sensitivity=[0.2, 0.1, 0.1],
                #     debug_vis=True,
                #     cmd_arrow_offset=[0, 0, 0.5],
                #     ranges=Se2KeyboardCommandCfg.Ranges(
                #         lin_vel_x=env_cfg.commands.base_velocity.ranges.lin_vel_x,
                #         lin_vel_y=env_cfg.commands.base_velocity.ranges.lin_vel_y,
                #         ang_vel_z=env_cfg.commands.base_velocity.ranges.ang_vel_z,
                #     ),
                # )
                # env_cfg.commands = env_cfg.commands.replace(base_velocity=KEYBOARD_CFG)
                break
            case "2":
                # GAMEPAD_CFG = Se2GamepadCommandCfg(
                #     asset=SceneEntityCfg("robot", body_names="body"),
                #     deadzone=0.1,
                #     debug_vis=True,
                #     ranges=Se2GamepadCommandCfg.Ranges(
                #         lin_vel_x=env_cfg.commands.base_velocity.ranges.lin_vel_x,
                #         lin_vel_y=env_cfg.commands.base_velocity.ranges.lin_vel_y,
                #         ang_vel_z=env_cfg.commands.base_velocity.ranges.ang_vel_z,
                #     ),
                #     axes_map=Se2GamepadCommandCfg.AxesMap(
                #         left_stick_down=("-", "x"),
                #         left_stick_up=("+", "x"),
                #         left_stick_right=("-", "y"),
                #         left_stick_left=("+", "y"),
                #         right_stick_right=("-", "z"),
                #         right_stick_left=("+", "z"),
                #     ),
                # )
                # env_cfg.commands = env_cfg.commands.replace(base_velocity=GAMEPAD_CFG)
                break
            case "3":
                # Uses Command Term defined in RLTaskEnvCfg
                break
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    # load the policy
    resume_path = ""
    env_cfg = None

    # load configuration
    # while True:
    #     policy_load_type = input(
    #         "Select a policy loading option: \n"
    #         + "(1) to load latest policy from local,\n"
    #         + "(2) to load from explicit path to policy .pt file,\n"
    #         + "(3) to load from Weights and Biases\n"
    #     )
    #     match policy_load_type:
    #         case "1":
    #             resume_path = get_checkpoint_path(
    #                 log_root_path, run_dir=".*", checkpoint="model_.*.pt", sort_alpha=True
    #             )
    #             env_cfg = cli_args.load_local_cfg(resume_path)
    #             break
    #         case "2":
    #             policy_path = input("Enter the path of the policy model file to play \n")
    #             resume_path = os.path.abspath(policy_path)
    #             env_cfg = cli_args.load_local_cfg(resume_path)
    #             break
    #         case "3":
    #             run_path = input(
    #                 "Enter the weights and biases run path located on the Overview panel; i.e usr/Spot-Blind/abc123 \n"
    #             )
    #             model_name = input("Enter the name of the model file to download; i.e model_100.pt \n")
    #             resume_path, env_cfg = cli_args.pull_policy_from_wandb(log_dir, run_path, model_name)
    #             break
    # if not os.path.exists(resume_path):
    #     raise FileNotFoundError
    

    # specify directory for logging runs: {time-stamp}_{run_name}

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)


    model_root_path = "/".join(resume_path.split("/")[:-1])
    print(f"[INFO]: Model_root_path: {model_root_path}")

    log_dir = os.path.join(model_root_path, "play")

    print(f"created log dir at {log_dir}")
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
        }
        print("[INFO] Recording policy evaluation rollout.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        env.metadata["video.frames_per_second"] = 1.0 / env.unwrapped.step_dt
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path, load_optimizer=False)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    model_file_name = os.path.splitext(os.path.basename(resume_path))[0]
    export_model_dir = os.path.join(log_dir, "exported")
    # export_policy_as_jit(ppo_runner.alg.actor_critic, export_model_dir, filename=f"{model_file_name}_jit.pt")

    # jit_policy = torch.jit.load(os.path.join(export_model_dir, f"{model_file_name}_jit.pt"))
    # jit_policy = jit_policy.to(env.device)

    if args_cli.convert_onnx:
        print(f"[INFO]: Saving env config json file to {export_model_dir}")
        cfg_save_path = os.path.join(export_model_dir, "env_cfg.json")
        with open(cfg_save_path, "w") as fp:
            print("env_cfg: ", env_cfg)
            json.dump(env_cfg, fp, indent=4)
        print(f"[INFO]: Saving policy onnx file to {export_model_dir}")
        export_policy_as_onnx(ppo_runner.alg.actor_critic, export_model_dir, filename=f"{model_file_name}_policy.onnx")

    # Parse env_config.json to set Spot's initial joint state
    spot = env.unwrapped.scene["robot"]
    joint_names = spot.joint_names

    # print the input device configuration
    if hasattr(env.unwrapped.command_manager._terms["base_velocity"], "input_device"):
        print("** Input Device Configuration **")
        print(env.unwrapped.command_manager._terms["base_velocity"].input_device)

    # get host device of sim
    device = env.unwrapped.device
    init_root_pos = env_cfg["scene"]["robot"]["init_state"]["pos"]
    init_joint_state = env_cfg["scene"]["robot"]["init_state"]["joint_pos"]
    # match regex in name with init_joint state
    _, _, init_joint_state_values = resolve_matching_names_values(init_joint_state, joint_names)
    if len(init_joint_state_values) != len(joint_names):
        raise ValueError("Unable to match all joints with regex in env_config data")
    # move to torch Tensors and copy default states to all robots in env
    init_joint_state = torch.Tensor(init_joint_state_values).to(device).unsqueeze(0)
    init_root_pos = torch.Tensor(init_root_pos).to(device).unsqueeze(0)
    spot.data.default_joint_pos = init_joint_state.repeat(args_cli.num_envs, 1)
    spot.data.defaut_pos = init_root_pos.repeat(args_cli.num_envs, 1)
    # reset environment
    obs, _ = env.get_observations()
    # simulate environment
    step_ctr = 0
    step_dt = env.unwrapped.step_dt
    with torch.no_grad():
        for i in range(500):
            # agent stepping
            env.unwrapped.command_manager.compute(step_dt)
            obs[:, 9:12] = env.unwrapped.command_manager.get_command("base_velocity")
            actions = jit_policy(obs)

            # env stepping
            obs, _, _, _ = env.step(actions)
            # env rendering
            env.unwrapped.render()
            # check if simulator is stopped
            if env.unwrapped.sim.is_stopped():
                break
            step_ctr += 1

        # close the simulator
        env.close()


if __name__ == "__main__":
    try:
        # run the main execution
        run_sim()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
