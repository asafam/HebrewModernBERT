#!/bin/bash
#SBATCH --job-name=hmb-lg-p2-g1
#SBATCH --output=.slurm/logs/hmb-lg-p2-g1_%j.out
#SBATCH --error=.slurm/logs/hmb-lg-p2-g1_%j.err
#SBATCH --partition=p_b200_goldberg
#SBATCH --account=ug_goldberg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --mail-user=achimoa@biu.ac.il
#SBATCH --mail-type=END,FAIL,REQUEUE

# LARGE phase-2 (anneal) 1-GPU FALLBACK for when the 4-GPU p_b200_nlp quota is blocked.
# Same checkpoints via autoresume; global_train_batch_size 576 just uses more grad accumulation.
# Slower but keeps training instead of idling -- switch back to train_b200_large_phase-2.slurm
# when 4 GPUs free up (docs/TRAINING.md §6).

# Run a SNAPSHOT of the wrapper, not the file in the repo. Bash reads a script incrementally from a
# byte offset, so editing the wrapper while a job is running corrupts that job's parse mid-execution
# (this killed smoke job 21316657 at its epilogue after a clean 60/60 run). Over a multi-week phase-0
# with ~33 requeues, editing scripts between requeues is normal -- snapshotting makes it safe.
SNAP="${SLURM_TMPDIR:-/tmp}/$(basename scripts/train_modernbert_large_phase-2-contextextension.sh).${SLURM_JOB_ID:-$$}"
cp "scripts/train_modernbert_large_phase-2-contextextension.sh" "$SNAP"

# --- wall-safe self-requeue -------------------------------------------------------------
# slurm's requeue-on-TIMEOUT is not reliable here (job 24881738 died TIMEOUT on its first
# block and left afterok dependents in DependencyNeverSatisfied, breaking the chain).
# --signal=B:USR1@180 delivers USR1 to THIS batch script 3 min before the wall; we requeue
# explicitly. The child must run in the background with `wait` or the trap cannot fire.
_wall_requeue() {
    echo "[wall in ~3min] requeueing job $SLURM_JOB_ID so the chain survives"
    scontrol requeue "$SLURM_JOB_ID"
}
trap _wall_requeue USR1

bash "$SNAP" &
CHILD=$!
wait $CHILD
RC=$?
rm -f "$SNAP"
exit $RC
