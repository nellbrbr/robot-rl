
# Training & Testing a Quadruped Locomotion Policy with Isaac Lab

This repo is a slightly modified version of [the official Isaac Lab repo](https://github.com/isaac-sim/IsaacLab), and the associated documentation is [here](https://isaac-sim.github.io/IsaacLab/#).

For instructions on deploying a trained Boston Dynamics Spot locomotion policy on a real robot, see [this technical blog](https://developer.nvidia.com/blog/closing-the-sim-to-real-gap-training-spot-quadruped-locomotion-with-nvidia-isaac-lab/).

## Installation
- Confirm that your system meets the minimum requirements for Isaac Sim as described [here](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/requirements.html). Note: It is recommended that your GPU has at least 16GB of VRAM for Isaac Lab.
- Install Isaac Sim and Isaac Lab following [these instructions](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html) (substitute **this** repo for the one specified).
- You may wish to follow [these instructions](https://isaac-sim.github.io/IsaacLab/source/setup/developer.html) to set up your development environment, but it should not be necessary.

## Locomotion Training with Boston Dynamics Spot

### To train a Spot locomotion policy:

```
cd <path-to-this-repo>
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Flat-Spot-v0 --num_envs 4096 --headless --video --enable_cameras --max_iterations 20000
```

### To evaluate the policy without keyboard control (command manager will randomly sample commands, and you will not be able to teleoperate the robot):

Change the value after --num_envs if you wish to play the policy with more than 1 robot. 

```
./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py --task Isaac-Velocity-Flat-Spot-Play-v0 --num_envs 1
```

### To evaluate the policy with keyboard control:

```
./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py --task Isaac-Velocity-Flat-Spot-Play-v0 --num_envs 1 --keyboard
```
### Key Bindings for Velocity Commands

|Command |Key (+ve axis) |Key (-ve axis) |
|--------|--------|--------|
| Move along x-axis | Numpad 8 / Arrow Up | Numpad 2 / Arrow Down |
| Move along y-axis | Numpad 4 / Arrow Right | Numpad 6 / Arrow Left|
| Rotate along z-axis | Numpad 7 / X | Numpad 9 / Y |

---

# Isaac Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.1-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)


**Isaac Lab** is a unified and modular framework for robot learning that aims to simplify common workflows
in robotics research (such as RL, learning from demonstrations, and motion planning). It is built upon
[NVIDIA Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html) to leverage the latest
simulation capabilities for photo-realistic scenes and fast and accurate simulation.

Please refer to our [documentation page](https://isaac-sim.github.io/IsaacLab) to learn more about the
installation steps, features, tutorials, and how to set up your project with Isaac Lab.

## Contributing to Isaac Lab

We wholeheartedly welcome contributions from the community to make this framework mature and useful for everyone.
These may happen as bug reports, feature requests, or code contributions. For details, please check our
[contribution guidelines](https://isaac-sim.github.io/IsaacLab/source/refs/contributing.html).

## Troubleshooting

Please see the [troubleshooting](https://isaac-sim.github.io/IsaacLab/source/refs/troubleshooting.html) section for
common fixes or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

For issues related to Isaac Sim, we recommend checking its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html)
or opening a question on its [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

* Please use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussing ideas, asking questions, and requests for new features.
* Github [Issues](https://github.com/isaac-sim/IsaacLab/issues) should only be used to track executable pieces of work with a definite scope and a clear deliverable. These can be fixing bugs, documentation issues, new features, or general updates.

## License

The Isaac Lab framework is released under [BSD-3 License](LICENSE). The license files of its dependencies and assets are present in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab development initiated from the [Orbit](https://isaac-orbit.github.io/) framework. We would appreciate if you would cite it in academic publications as well:

```
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal={IEEE Robotics and Automation Letters},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year={2023},
   volume={8},
   number={6},
   pages={3740-3747},
   doi={10.1109/LRA.2023.3270034}
}
```
