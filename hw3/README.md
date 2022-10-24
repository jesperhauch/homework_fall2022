# Q-learning
## Question 1: Basic Q-Learning Performance (DQN)
```
python cs285/scripts/run_hw3_dqn.py --env_name MsPacman-v0 --exp_name q1
```

## Question 2: double Q-learning (DDQN)

```python
import shlex, subprocess, cryptography
commands = ["python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_1 --seed 1",
            "python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_2 --seed 2",
            "python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_3 --seed 3",
            "python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_1 --double_q --seed 1",
            "python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_2 --double_q --seed 2",
            "python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_3 --double_q --seed 3"]

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.Popen(args)
```

## Question 3: Experimenting with Hyperparameters
Since the learning rate argument could not be passed to `run_hw3_dqn.py`, I had to manually change the default learning rate for LunarLander in `dqn_utils.py` Line 167. I ran 3 separate runs where I fixed the learning rate to 0.0005, 0.005, and 0.01 respectively. I ran the line below but with different experiment name depending on the learning rate.
```
python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q3_lr_<> --seed 1
```

# Actor-critic
## Question 4: Sanity check with Cartpole
```python
import shlex, subprocess, cryptography
commands = ["python cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_ac_1_1 -ntu 1 -ngsptu 1",
            "python cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_100_1 -ntu 100 -ngsptu 1",
            "python cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_1_100 -ntu 1 -ngsptu 100",
            "python cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_10_10 -ntu 10 -ngsptu 10"]

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.Popen(args)
```

## Question 5: Run actor-critic with more difficult tasks
InvertedPendulum-v4
```
python cs285/scripts/run_hw3_actor_critic.py --env_name InvertedPendulum-v4 --ep_len 1000
--discount 0.95 -n 100 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name q5_InvertedPendulum-v2_1_100 -ntu 1
-ngsptu 100
```

HalfCheetah-v4
```
python cs285/scripts/run_hw3_actor_critic.py --env_name HalfCheetah-v4 --ep_len 150 --
discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --
exp_name q5_HalfCheetah-v4_1_100 -ntu 1 -ngsptu 100
```

# Soft actor-critic
## Question 6: 
InvertedPendulum-4
```
python cs285/scripts/run_hw3_sac.py \
--env_name InvertedPendulum-v4 --ep_len 1000 \
--discount 0.99 --scalar_log_freq 1000 \
-n 100000 -l 2 -s 256 -b 1000 -eb 2000 \
-lr 0.0003 --init_temperature 0.1 --exp_name q6a_sac_InvertedPendulum \
--seed 1
```

HalfCheetah-4
```
python cs285/scripts/run_hw3_sac.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 1500 \
-n 200000 -l 2 -s 256 -b 1500 -eb 1500 \
-lr 0.0003 --init_temperature 0.1 --exp_name q6b_sac_HalfCheetah \
--seed 1 --actor_update_frequency 10 -tb 1500
```