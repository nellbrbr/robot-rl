import argparse
from omni.isaac.lab.app import AppLauncher
import cli_args


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps)")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations"
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--policy_path", type=str, default=None, help="Path to the policy to use in simulation")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import torch
import pika
import sys

from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.utils.dict import print_dict

from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper
)

def format_action(action_raw):
    """
    Turns RabbitMQ message body into a tensor of floats to pass to the policy

    Args:
        action_raw (str) : message body to be converted; expected in form like '0.1,0.2,0.3'

    Returns:
        (torch.Tensor) : tensor-fied x,y,heading for policy
    """
    c_split = action_raw.split(',')
    c_list = [float(t) for t in c_split]
    return torch.Tensor([c_list])


class InteractiveSimulator:

    def __init__(self, parser):
        self.actions = torch.Tensor([[0.0,0.0,0.0]])
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))

    def set_actions(self, actions):
        """
        actions = x,y,heading from RabbitMQ msg 
        """
        self.actions = format_action(actions)

    def get_actions(self):
        return self.actions

    def main(self):
        channel = self.connection.channel()
        channel.queue_declare(queue='hello')
        # parse configuration
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        )
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during simulation.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)
        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(env)

        if args_cli.policy_path is not None:
            resume_path = args_cli.policy_path

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)

        # obtain the trained policy for inference
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

        def callback(ch, method, properties, body): #called when a message is received 
            print(f" [x] Received {body}")
            body = body.decode()
            self.set_actions(body) #here, we set the actions using the x,y,heading from the message body
            with torch.inference_mode():
                obs, _, _, _ = env.step(self.get_actions()) #which means this only happens when a message is received

        channel.basic_consume(queue='hello', on_message_callback=callback, auto_ack=True)

        print(' [*] Waiting for messages. To exit press CTRL+C')
        channel.start_consuming()

        # reset environment
        obs, _ = env.get_observations()




if __name__ == '__main__':
    app = InteractiveSimulator(parser)
    try:
        app.main()
    except KeyboardInterrupt:
        print('Interrupted')
        simulation_app.close()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


