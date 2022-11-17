import shlex, subprocess

command_stem = [
"python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --unsupervised_exploration --awac_lambda={l} --exp_name q4_awac_medium_unsupervised_lam{l}",
"python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000                            --awac_lambda={l} --exp_name q4_awac_medium_supervised_lam{l}",
]

awac_l = [0.1,1,2,10,20,50]

commands = []
for command in command_stem:
    for l in awac_l:
        commands.append(command.format(l=l))

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.run(args)