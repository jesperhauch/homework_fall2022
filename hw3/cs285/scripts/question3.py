import shlex, subprocess, cryptography
commands = ["python cs285/scripts/run_hw3_dqn.py --env_name MsPacman-v0 --exp_name q1", 
            "python cs285/scripts/run_hw3_dqn.py --env_name MsPacman-v0 --exp_name q3_hparam1", 
            "python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q3_hparam2", 
            "python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q3_hparam3"]

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.Popen(args)