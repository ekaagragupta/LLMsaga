
checkpoint=$1

echo "Running consolidation with checkpoint: $checkpoint"

export SCRATCH=/scratch/${USER}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-$SCRATCH/checkpoints}
export EVAL_DIR=${EVAL_DIR:-$SCRATCH/workspace/mllm_eval_hpc}


singularity exec --bind /scratch --nv --overlay $SCRATCH/overlay/cambrian.ext3:ro $SCRATCH/overlay/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "
source $SCRATCH/env.sh

cd $EVAL_DIR
export LLM_MODEL_NAME=NousResearch/Nous-Hermes-2-Yi-34B
bash scripts/consolidate_gcp.bash $checkpoint
"