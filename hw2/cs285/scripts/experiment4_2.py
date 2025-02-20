import shlex, subprocess, cryptography
batch_size = 30000
lr = 0.02
exp_1= "python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b {batch_size} -lr {lr} \
--exp_name q4_b{batch_size}_r{lr}".format(batch_size=batch_size, lr=lr)
exp_2 = "python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b {batch_size} -lr {lr} -rtg \
--exp_name q4_b{batch_size}_r{lr}_rtg".format(batch_size=batch_size, lr=lr)
exp_3 = "python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b {batch_size} -lr {lr} --nn_baseline \
--exp_name q4_b{batch_size}_r{lr}_nnbaseline".format(batch_size=batch_size, lr=lr)
exp_4 = "python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b {batch_size} -lr {lr} -rtg --nn_baseline \
--exp_name q4_b{batch_size}_r{lr}_rtg_nnbaseline".format(batch_size=batch_size, lr=lr)
commands = [exp_1, exp_2, exp_3, exp_4]

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.run(args)
