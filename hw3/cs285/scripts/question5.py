import shlex, subprocess, cryptography
commands = [
    "python cs285/scripts/run_hw3_actor_critic.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.95 -n 100 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name q5_IP_1_100 -ntu 1 -ngsptu 100",
    "python cs285/scripts/run_hw3_actor_critic.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name q5_HC_1_100 -ntu 1 -ngsptu 100"]

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        proc = subprocess.Popen(args)
        #proc.kill()