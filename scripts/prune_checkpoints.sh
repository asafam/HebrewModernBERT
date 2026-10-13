#!/bin/bash
# Prune intermediate Composer checkpoints, keeping only what is still needed.
#
# WHY THERE ARE SO MANY. The yamls set save_num_checkpoints_to_keep: 20, but Composer's pruning
# tracker only knows about checkpoints IT saved in the current session. Every 4h-wall requeue
# starts a fresh session, and a session saves ~5 checkpoints before the wall -- never reaching
# 20, so nothing is ever pruned. Phase-0 alone accumulated 111 files over 21 sessions.
# checkpoints/large is ~1.3TB on shared storage as a result.
#
# WHAT IS KEPT per phase:
#   * latest-rank0.pt and whatever it points at  -- the phase's final weights
#   * any checkpoint that was converted to HF and gated (so a published number stays traceable
#     to a file that still exists)
# WHAT IS DELETED: every other ep0-ba*.pt.
#
# This is safe because the FINAL models already exist as HF exports (outputs/hf/...), which is
# what anyone needs to use or publish them. Composer checkpoints only matter for RESUMING
# training, and all three phases are complete.
#
# DRY RUN BY DEFAULT. It prints what it would delete and the space it would reclaim; it deletes
# nothing unless you pass --delete.
#
#   bash scripts/prune_checkpoints.sh            # list only
#   bash scripts/prune_checkpoints.sh --delete   # actually remove
set -euo pipefail
cd /home/nlp/achimoa/workspace/HebrewModernBERT

DELETE=0; [ "${1:-}" = "--delete" ] && DELETE=1

# Checkpoints that were gated/converted and should survive for traceability.
KEEP_EXTRA="ep0-ba36000-rank0.pt ep0-ba222000-rank0.pt ep0-ba6000-rank0.pt ep0-ba5000-rank0.pt"

total_freed=0
for d in checkpoints/large/*/ckpt checkpoints/base/*/ckpt; do
    [ -d "$d" ] || continue
    # resolve latest-rank0.pt -> the file it points at
    keep=""
    if [ -L "$d/latest-rank0.pt" ]; then
        keep="$(readlink "$d/latest-rank0.pt")"
    fi
    echo "=== $d"
    [ -n "$keep" ] && echo "    KEEP (latest): $keep"
    n_del=0; sz_del=0
    for f in "$d"/ep0-ba*.pt; do
        [ -e "$f" ] || continue
        b="$(basename "$f")"
        [ "$b" = "$keep" ] && continue
        skip=0
        for k in $KEEP_EXTRA; do [ "$b" = "$k" ] && { echo "    KEEP (gated):  $b"; skip=1; break; }; done
        [ "$skip" = "1" ] && continue
        sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
        n_del=$((n_del+1)); sz_del=$((sz_del+sz))
        [ "$DELETE" = "1" ] && rm -f "$f"
    done
    total_freed=$((total_freed+sz_del))
    printf "    %s %d files, %.1f GB\n" "$([ "$DELETE" = 1 ] && echo DELETED || echo 'would delete')" \
           "$n_del" "$(echo "$sz_del" | awk '{print $1/1073741824}')"
done

printf "\nTOTAL %s: %.1f GB\n" "$([ "$DELETE" = 1 ] && echo freed || echo reclaimable)" \
       "$(echo "$total_freed" | awk '{print $1/1073741824}')"
if [ "$DELETE" != "1" ]; then
    echo
    echo "DRY RUN -- nothing deleted. Re-run with --delete to remove."
    echo "The final models are already exported to outputs/hf/ and are not touched by this script."
fi
