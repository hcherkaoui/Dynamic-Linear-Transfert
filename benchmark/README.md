# Usage (Jean Zay)

## 1) Install
```bash
 bash setup.sh
```

## 2) Interactive session (to be skipped if you are not on Jean Zhay)
```bash
srun --pty -A jcx@v100 -p gpu_p13 --gres=gpu:1 -c 8 --time=01:00:00 bash
```

## 2) Submit the benchmark job (compute node via Slurm)
```bash
bash launcher.sh
```