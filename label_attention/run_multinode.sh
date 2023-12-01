#PBS -N biosift
#PBS -l select=2:ncpus=30:ngpus=2:gpu_model=a100:mem=200gb,walltime=72:00:00
#PBS -j oe
#PBS -o biobert.log

cd $PBS_O_WORKDIR

### JOB PARAMS
model_name=$MODEL_NAME
dataset="/scratch/taw2/biosift/dataset/hf_datasets/binary_dataset/"
ENV_NAME="/scratch/taw2/conda_envs/biosift_env"

NGPUS=2 # per node
LAUNCH_SCRIPT="${PBS_O_WORKDIR}/run.sh"

# Get optimal hyperparameters from raytune output
# TUNE_DIR="/scratch/taw2/.cache/raytune/"
# write_file="/scratch/taw2/biosift/benchmarks/supervised/output/hyperparameters/${model//\//_}.json"
# TARGET_DIR="${TUNE_DIR}${model//\//_}_supervised_pbt"
# find $TARGET_DIR -name ".wandb" -prune -o -type d | xargs -n 1 -I {} find {} -type f -name result.json | xargs cat | jq -s 'sort_by(.objective)' | jq '.[-1]' > $write_file


# Get optimal hyperparameters from config
hp_config="${PBS_O_WORKDIR}/output/hyperparameters/${model_name//\//_}.json"
batch_size=$( cat $hp_config | jq .config.per_device_train_batch_size )
weight_decay=$( cat $hp_config | jq .config.weight_decay )
learning_rate=$( cat $hp_config | jq .config.learning_rate )

# timestamp output directory name
# and create directory.
timestamp=$(date +%D_%H_%M_%S | tr / _)
OUTPUT_DIR="${PBS_O_WORKDIR}/output/run_data/${model_name//\//_}_${timestamp}"
# mkdir -p $OUTPUT_DIR

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
TOTAL_PROCS=$(( NNODES * NGPUS )) 

export WANDB_PROJECT="biosift"
export WANDB_JOB_NAME="supervised_${model_name//\//_}"

pbsdsh -- bash "${PBS_O_WORKDIR}/run.sh" \
        $HOSTNAME \
        $ENV_NAME \
        $NNODES \
        $NGPUS \
        "./src/run_supervised.py \
                --model_name_or_path ${model_name} \
                --dataset_name ${dataset} \
                --shuffle_train_dataset \
                --text_column_name Abstract \
                --do_train \
                --do_eval \
                --do_predict \
                --max_seq_length 512 \
                --per_device_train_batch_size $(( $batch_size / $TOTAL_PROCS )) \
                --learning_rate $learning_rate \
                --weight_decay $weight_decay \
                --num_train_epochs 5 \
                --save_total_limit 1 \
		--save_strategy epoch \
                --evaluation_strategy epoch \
                --load_best_model_at_end \
                --output_dir ${OUTPUT_DIR}/" 

        
        


