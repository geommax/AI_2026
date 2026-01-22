## source
```
source ~/.bashrc_conda
conda activate env_isaaclab
```

## how to train g1
```
cd /home/mr_robot/ISAAAC/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1
# copy Isaac-Velocity-Rough-G1-v0 from __init__.py
```

#### train g1
```
cd /home/mr_robot/ISAAAC/IsaacLab/scripts/reinforcement_learning/rsl_rl
python train.py --task=Isaac-Velocity-Rough-G1-v0 --num_envs=64 --headless
# check output in /home/mr_robot/ISAAAC/IsaacLab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl
```
#### play g1
```
cd /home/mr_robot/ISAAAC/IsaacLab/scripts/reinforcement_learning/rsl_rl
python play.py --task=Isaac-Velocity-Rough-G1-v0 --num_envs=5 checkpoint=logs/rsl_rl/g1_rough/2026-01-22_20-20-51/model_2999.pt
# or use 
# +checkpoint=logs/rsl_rl/g1_rough/2026-01-22_20-20-51/model_2999.pt
```

## how to train go2
```
cd /home/mr_robot/ISAAAC/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2
# copy Isaac-Velocity-Rough-Unitree-Go2-v0 from __init__.py
```

#### train go2
```
cd /home/mr_robot/ISAAAC/IsaacLab/scripts/reinforcement_learning/rsl_rl
python train.py --task=Isaac-Velocity-Rough-Unitree-Go2-v0 --num_envs=128 --headless
# check output in /home/mr_robot/ISAAAC/IsaacLab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl
```
#### play go2
```
cd /home/mr_robot/ISAAAC/IsaacLab/scripts/reinforcement_learning/rsl_rl
python play.py --task=Isaac-Velocity-Rough-Unitree-Go2-v0 --num_envs=5 +checkpoint=logs/rsl_rl/unitree_go2_rough/2026-01-23_03-41-53/model_1499.pt
```