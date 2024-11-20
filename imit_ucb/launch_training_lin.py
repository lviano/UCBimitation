import os
for seed in range(10):
    print(seed)
    os.system(f"python train_learner/ilarl_lin.py --env-name LinMDP-v0 --num-threads 1 --seed {seed} --beta 1e-2 --max-iter-num 20 --n-expert-trajs 1")
    #os.system(f"python train_learner/fra_lin.py --env-name LinMDP-v0 --num-threads 1 --seed {seed} --beta 1e-2 --max-iter-num 400 --n-expert-trajs 1")
    os.system(f"python train_learner/ilarl_lin.py --env-name LinMDP-v0 --num-threads 1 --seed {seed} --beta 1e-2 --max-iter-num 20 --n-expert-trajs 10")
    os.system(f"python train_learner/ilarl_lin.py --env-name LinMDP-v0 --num-threads 1 --seed {seed} --beta 1e-2 --max-iter-num 20 --n-expert-trajs 100")
    os.system(f"python train_learner/ppil_lin.py --env-name LinMDP-v0 --num-threads 1 --seed {seed} --max-iter-num 80 --n-expert-trajs 1")
    os.system(f"python train_learner/ppil_lin.py --env-name LinMDP-v0 --num-threads 1 --seed {seed} --max-iter-num 80 --n-expert-trajs 10")
    os.system(f"python train_learner/ppil_lin.py --env-name LinMDP-v0 --num-threads 1 --seed {seed} --max-iter-num 80 --n-expert-trajs 100")
    os.system(f"python train_learner/iqlearn_lin.py --env-name LinMDP-v0 --num-threads 1 --seed {seed} --max-iter-num 80 --n-expert-trajs 1")
    os.system(f"python train_learner/iqlearn_lin.py --env-name LinMDP-v0 --num-threads 1 --seed {seed} --max-iter-num 80 --n-expert-trajs 10")
    os.system(f"python train_learner/iqlearn_lin.py --env-name LinMDP-v0 --num-threads 1 --seed {seed} --max-iter-num 80 --n-expert-trajs 100")

    