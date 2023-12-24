# CS7309_RL_Project

## Overview

这是强化学习理论与算法Final Project仓库。本次项目在Atari和MuJoco环境中分别使用了value-based 和 policy-based算法进行测试。具体地，对于Atari环境，使用DQN算法分别对VideoPinball, Breakout, Pong, Boxing进行了测试。对于MuJoco环境，使用PPO和TD3算法分别对Hopper, Humanoid, HalfCheetah, Ant进行了测试。

## Environment Setup

首先创建conda环境，并安装PyTorch，得益于PyTorch的高版本兼容性，本次实验使用最新版Torch环境
```shell
conda create -n RL_3.9 python=3.9
conda activate RL_3.9
pip install torch torchvision # pip install torch==2.1.2 torchvision==0.16.2
```

然后安装gym环境，由于不同版本间API差距较大，本次实验使用`gym==0.21.0`
```shell
pip install gym==0.21.0
pip install ale_py==0.7.5
```
如果遇到安装问题，可以参考Troubleshooting部分。

然后执行以下脚本安装剩余依赖文件,
```shell
pip install -r requirements.txt
```

## File Structure



## Troubleshooting

### 1. gym安装问题
安装时如果遇到以下问题
```shell
Collecting gym==0.21.0
  Using cached gym-0.21.0.tar.gz (1.5 MB)
  Preparing metadata (setup.py) ... error
  error: subprocess-exited-with-error
 
  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [1 lines of output]
      error in gym setup command: 'extras_require' must be a dictionary whose values are strings or lists of strings containing valid project/version requirement specifiers.
      [end of output]
 
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed
 
× Encountered error while generating package metadata.
╰─> See above for output.
 
note: This is an issue with the package mentioned above, not pip.
hint: See above for details.
```

则需要升级`setuptools`和`wheel`版本
```shell
pip install --upgrade setuptools
pip install --user --upgrade wheel
```

### 2. gym无法找到环境
如果执行脚本时遇到以下问题
```shell
gym.error.Error: We're Unable to find the game "Breakout". Note: Gym no longer distributes ROMs. If you own a license to use the necessary ROMs for research purposes you can download them via `pip install gym[accept-rom-license]`. Otherwise, you should try importing "Breakout" via the command `ale-import-roms`. 
```

则需要执行以下命令
```shell
pip install gym[accept-rom-license]
```