#!/bin/bash
#SBATCH --job-name=hmb-lg-smk1-g1
#SBATCH --output=.slurm/logs/hmb-lg-smk1-g1_%j.out
#SBATCH --error=.slurm/logs/hmb-lg-smk1-g1_%j.err
#SBATCH --partition=p_b200_goldberg
#SBATCH --account=ug_goldberg  # both p_b200_* partitions require an explicit account
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1    # one task; Composer spawns the GPU processes (DDP) itself
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=01:00:00
#SBATCH --mail-user=achimoa@biu.ac.il
#SBATCH --mail-type=END,FAIL

# LARGE phase-1 (8192, packed) microbatch probe on a SINGLE B200 -- validate WITHOUT DDP first,
# per docs/TRAINING.md §8. At 8192 the failure can be an illegal memory access (flash-attn triton
# rotary int32 limit) rather than a clean OOM, so this starts at MB=8 instead of auto-searching.
#   sbatch --export=ALL,MB=12 .slurm/jobs/smoke_b200_large_phase-1_1gpu.slurm

# Run a SNAPSHOT of the wrapper, not the file in the repo. Bash reads a script incrementally from a
# byte offset, so editing the wrapper while a job is running corrupts that job's parse mid-execution
# (this killed smoke job 21316657 at its epilogue after a clean 60/60 run). Over a multi-week phase-0
# with ~33 requeues, editing scripts between requeues is normal -- snapshotting makes it safe.
SNAP="${SLURM_TMPDIR:-/tmp}/$(basename scripts/smoke_test_large_phase-1_b200.sh).${SLURM_JOB_ID:-$$}"
cp "scripts/smoke_test_large_phase-1_b200.sh" "$SNAP"
bash "$SNAP"
RC=$?
rm -f "$SNAP"
exit $RC
