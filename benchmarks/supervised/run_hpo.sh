#PBS -N biosift
#PBS -l select=1:ncpus=24:ngpus=2:gpu_model=v100:mem=100gb,walltime=72:00:00
#PBS -j oe
#PBS -o output/supervised_run.log

cd $PBS_O_WORKDIR

### JOB PARAMS
model_name="monologg/biobert_v1.1_pubmed"
dataset="/scratch/taw2/biosift/dataset/hf_datasets/binary_dataset/"
ENV_NAME="/scratch/taw2/conda_envs/biosift_env"

NGPUS=2

# timestamp output directory name
# and create directory.
timestamp=$(date +%D_%H_%M_%S | tr / _)
OUTPUT_DIR="${PBS_O_WORKDIR}/output/${model_name//\//_}_hpo_${timestamp}"
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
export WANDB_LOG_MODEL=true
export WANDB_JOB_NAME="supervised_hpo_${model_name//\//_}"

"./src/run_supervised_hpo.py \
        --model_name_or_path ${model_name} \
        --dataset_name ${dataset} \
        --shuffle_train_dataset \
        --text_column_name Abstract \
        --do_train \
        --do_eval \
        --do_predict \
        --max_seq_length 512 \
        --per_device_train_batch_size 4 \
        --learning_rate 2e-5 \
        --num_train_epochs 5 \
        --save_strategy epoch \
        --evaluation_strategy epoch \
        --load_best_model_at_end \
        --output_dir ${output_dir}/" 

