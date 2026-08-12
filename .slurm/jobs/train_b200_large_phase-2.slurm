#!/bin/bash
#SBATCH --job-name=hmb-lg-p2
#SBATCH --output=.slurm/logs/hmb-lg-p2_%j.out
#SBATCH --error=.slurm/logs/hmb-lg-p2_%j.err
#SBATCH --partition=p_b200_nlp
#SBATCH --account=ug_goldberg  # both p_b200_* partitions require an explicit account
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1    # one task; Composer spawns the GPU processes (DDP) itself
#SBATCH --cpus-per-task=32
#SBATCH --mem=480GB
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --mail-user=achimoa@biu.ac.il
#SBATCH --mail-type=END,FAIL,REQUEUE

# LARGE phase-2: curated-heavy anneal at 8192 (15B tokens, one_minus_sqrt LR -> ~0).

# Run a SNAPSHOT of the wrapper, not the file in the repo. Bash reads a script incrementally from a
# byte offset, so editing the wrapper while a job is running corrupts that job's parse mid-execution
# (this killed smoke job 21316657 at its epilogue after a clean 60/60 run). Over a multi-week phase-0
# with ~33 requeues, editing scripts between requeues is normal -- snapshotting makes it safe.
SNAP="${SLURM_TMPDIR:-/tmp}/$(basename scripts/train_modernbert_large_phase-2-contextextension.sh).${SLURM_JOB_ID:-$$}"
cp "scripts/train_modernbert_large_phase-2-contextextension.sh" "$SNAP"
bash "$SNAP"
RC=$?
rm -f "$SNAP"
exit $RC
