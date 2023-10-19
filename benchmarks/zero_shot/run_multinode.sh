#PBS -N biosift
#PBS -l select=2:ncpus=24:ngpus=2:gpu_model=v100:mem=100gb,walltime=72:00:00
#PBS -j oe
#PBS -o output/zero_shot_run.log

cd $PBS_O_WORKDIR

### JOB PARAMS
model_name="monologg/biobert_v1.1_pubmed"
dataset="/scratch/taw2/biosift/dataset/hf_datasets/binary_dataset/"
ENV_NAME="/scratch/taw2/conda_envs/biosift_env"

NGPUS=2
LAUNCH_SCRIPT="${PBS_O_WORKDIR}/run.sh"

# timestamp output directory name
# and create directory.
timestamp=$(date +%D_%H_%M_%S | tr / _)
OUTPUT_DIR="${PBS_O_WORKDIR}/output/${model_name//\//_}_${timestamp}"
mkdir -p $OUTPUT_DIR

# Useful for distributed debugging
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1 

# Add modules (implementation dependent)
module add cuda/11.6.2-gcc/9.5.0
module add nccl/2.11.4-1-gcc/9.5.0-cu11_6-nvP-nvV-nvA
module add openmpi/4.1.3-gcc/9.5.0-cu11_6-nvP-nvV-nvA-ucx
module add anaconda3/2022.05-gcc/9.5.0

# Activate specified environment
source activate $ENV_NAME

# Get number of nodes. This will be the same as specified above.
NNODES=$(cat $PBS_NODEFILE | wc -l)
ncpus=$NCPUS

export WANDB_PROJECT="biosift"
export WANDB_JOB_NAME="zero_shot_${model_name//\//_}"

pbsdsh -- bash "${PBS_O_WORKDIR}/run.sh" \
        $HOSTNAME \
        $ENV_NAME \
        $NNODES \
        $NGPUS \
        "./src/run_zero_shot.py \
                --model_name_or_path ${model_name} \
                --dataset_name ${dataset} \
                --premise_column_name Abstract \
                --hypothesis_column_name Hypothesis
                --compute_threshold \
                --do_predict \
                --max_seq_length 512 \
                --output_dir ${output_dir}/" 

