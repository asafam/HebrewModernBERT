#!/bin/bash
#SBATCH --job-name=hmb-lg-smk0
#SBATCH --output=.slurm/logs/hmb-lg-smk0_%j.out
#SBATCH --error=.slurm/logs/hmb-lg-smk0_%j.err
#SBATCH --partition=p_b200_nlp
#SBATCH --account=ug_goldberg  # both p_b200_* partitions require an explicit account
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1    # one task; Composer spawns the GPU processes (DDP) itself
#SBATCH --cpus-per-task=32
#SBATCH --mem=480GB
#SBATCH --time=01:00:00
#SBATCH --mail-user=achimoa@biu.ac.il
#SBATCH --mail-type=END,FAIL

# LARGE phase-0 smoke on 4 B200s -- run this only for the REAL multi-GPU throughput number used to
# size the 130B-token phase-0. The node is often fully allocated, so this PENDs until 4 GPUs free
# up; the 1-GPU variant is the one to run first.

# Run a SNAPSHOT of the wrapper, not the file in the repo. Bash reads a script incrementally from a
# byte offset, so editing the wrapper while a job is running corrupts that job's parse mid-execution
# (this killed smoke job 21316657 at its epilogue after a clean 60/60 run). Over a multi-week phase-0
# with ~33 requeues, editing scripts between requeues is normal -- snapshotting makes it safe.
SNAP="${SLURM_TMPDIR:-/tmp}/$(basename scripts/smoke_test_large_phase-0_b200.sh).${SLURM_JOB_ID:-$$}"
cp "scripts/smoke_test_large_phase-0_b200.sh" "$SNAP"
bash "$SNAP"
RC=$?
rm -f "$SNAP"
exit $RC
