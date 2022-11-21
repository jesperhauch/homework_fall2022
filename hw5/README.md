## Part 1: "Unsupervised" RND and exploration performance
### First sub-part
```
python cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 \
--use_rnd --unsupervised_exploration --exp_name q1_env1_rnd

python cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 \
--unsupervised_exploration --exp_name q1_env1_random

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
--unsupervised_exploration --exp_name q1_env2_rnd

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 \
--unsupervised_exploration --exp_name q1_env2_random
```

### Second sub-part
```
python cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 \
--unsupervised_exploration --use_boltzmann --exp_name q1_alg_easy

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 \
--unsupervised_exploration --use_boltzmann --exp_name q1_alg_med
```

## Part 2: Offline learning on exploration data

### First sub-part
```
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_dqn \
--use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha 0

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_cql \
--use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha 0.1

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_cql_shift1_scale100 \
--use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha 0.1 \
--exploit_rew_shift 1 --exploit_rew_scale 100
```

### Second sub-part
Does shift and scale work better for CQL? If so, use in CQL runs below.
```
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps 5000 --offline_exploitation --cql_alpha 0.1 \
--unsupervised_exploration --exp_name q2_cql_numsteps_5000

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps 15000 --offline_exploitation --cql_alpha 0.1 \
--unsupervised_exploration --exp_name q2_cql_numsteps_15000

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps 5000 --offline_exploitation --cql_alpha 0.0 \
--unsupervised_exploration --exp_name q2_dqn_numsteps_5000

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps 15000 --offline_exploitation --cql_alpha 0.0 \
--unsupervised_exploration --exp_name q2_dqn_numsteps_15000
```

### Third sub-part

```
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
--unsupervised_exploration --offline_exploitation --cql_alpha 0.02 \
--exp_name q2_alpha0.02

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
--unsupervised_exploration --offline_exploitation --cql_alpha 0.5 \
--exp_name q2_alpha0.5
```

## Part 3: "Supervised" exploration with mixed reward bonuses

```
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps 20000 --cql_alpha 0.0 --exp_name q3_medium_dqn

python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps 20000 --cql_alpha 1.0 --exp_name q3_medium_cql

python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd \
--num_exploration_steps 20000 --cql_alpha 0.0 --exp_name q3_hard_dqn

python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd \
--num_exploration_steps 20000 --cql_alpha 1.0 --exp_name q3_hard_cql
```

## Part 4: Offline Learning with AWAC
Different lambdas using unsupervised exporation on PointmassEasy.
```
python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 \
--exp_name q4_awac_easy_unsupervised_lam0.1 --use_rnd --num_exploration_steps=20000 \
--unsupervised_exploration --awac_lambda=0.1

python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 \
--exp_name q4_awac_easy_unsupervised_lam1 --use_rnd --num_exploration_steps=20000 \
--unsupervised_exploration --awac_lambda=1

python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 \
--exp_name q4_awac_easy_unsupervised_lam2 --use_rnd --num_exploration_steps=20000 \
--unsupervised_exploration --awac_lambda=2

python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 \
--exp_name q4_awac_easy_unsupervised_lam10 --use_rnd --num_exploration_steps=20000 \
--unsupervised_exploration --awac_lambda=10

python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 \
--exp_name q4_awac_easy_unsupervised_lam20 --use_rnd --num_exploration_steps=20000 \
--unsupervised_exploration --awac_lambda=20

python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 \
--exp_name q4_awac_easy_unsupervised_lam50 --use_rnd --num_exploration_steps=20000 \
--unsupervised_exploration --awac_lambda=50
```
Different lambdas without unsupervised exploration on PointmassEasy
```
python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd \
--num_exploration_steps=20000 --awac_lambda=0.1 --exp_name q4_awac_easy_supervised_lam0.1

python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd \
--num_exploration_steps=20000 --awac_lambda=1 --exp_name q4_awac_easy_supervised_lam1

python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd \
--num_exploration_steps=20000 --awac_lambda=2 --exp_name q4_awac_easy_supervised_lam2

python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd \
--num_exploration_steps=20000 --awac_lambda=10 --exp_name q4_awac_easy_supervised_lam10

python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd \
--num_exploration_steps=20000 --awac_lambda=20 --exp_name q4_awac_easy_supervised_lam20

python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd \
--num_exploration_steps=20000 --awac_lambda=50 --exp_name q4_awac_easy_supervised_lam50
```

Different lambdas using unsupervised exporation on PointmassMedium.
```
python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \
--exp_name q4_awac_medium_unsupervised_lam0.1 --use_rnd --num_exploration_steps=20000 \
--unsupervised_exploration --awac_lambda=0.1

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \
--exp_name q4_awac_medium_unsupervised_lam1 --use_rnd --num_exploration_steps=20000 \
--unsupervised_exploration --awac_lambda=1

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \
--exp_name q4_awac_medium_unsupervised_lam2 --use_rnd --num_exploration_steps=20000 \
--unsupervised_exploration --awac_lambda=2

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \
--exp_name q4_awac_medium_unsupervised_lam10 --use_rnd --num_exploration_steps=20000 \
--unsupervised_exploration --awac_lambda=10

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \
--exp_name q4_awac_medium_unsupervised_lam20 --use_rnd --num_exploration_steps=20000 \
--unsupervised_exploration --awac_lambda=20

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \
--exp_name q4_awac_medium_unsupervised_lam50 --use_rnd --num_exploration_steps=20000 \
--unsupervised_exploration --awac_lambda=50
```
Different lambdas without unsupervised exploration on PointmassMedium
```
python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps=20000 --awac_lambda=0.1
--exp_name q4_awac_medium_supervised_lam0.1

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps=20000 --awac_lambda=1
--exp_name q4_awac_medium_supervised_lam1

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps=20000 --awac_lambda=2
--exp_name q4_awac_medium_supervised_lam2

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps=20000 --awac_lambda=10
--exp_name q4_awac_medium_supervised_lam10

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps=20000 --awac_lambda=20
--exp_name q4_awac_medium_supervised_lam20

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps=20000 --awac_lambda=50
--exp_name q4_awac_medium_supervised_lam50
```

## Part 5: Offline Learning with IQL

```
python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 \
--exp_name q5_iql_easy_supervised_lam0.1_tau0.5 --use_rnd \
--num_exploration_steps=20000 \
--awac_lambda=0.1 \
--iql_expectile=0.5
```

## Submitting
```
zip -vr submit.zip . -x "*.png"
```