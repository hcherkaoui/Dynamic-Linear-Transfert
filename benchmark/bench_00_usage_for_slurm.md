# Usage (Jean Zay)

## 1) One-time install (login node)
```bash
module purge
export WORK=/lustre/fswork/projects/rech/jcx/uor49lv
bash bench_01_install_for_slurm.sh
```

## 2) Submit the benchmark job (compute node via Slurm)
```bash
export WORK=/lustre/fswork/projects/rech/jcx/uor49lv
sbatch bench_02_run_all.slurm
```

## 3) Monitor
Follow your job' status:
```bash
squeue -u $USER
```
Follow logs (replace `<JOBID>`):
```bash
tail -f dyntr_bench_<JOBID>.out
tail -f dyntr_bench_<JOBID>.err
```

## 4) Cancel (if needed)
Stop it:
```bash
scancel <JOBID>
```
