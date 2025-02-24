HDFS_HOME=/data/alan-g491/simpleRL-reason/train
RUN_NAME=Qwen2.5-Math-7B_ppo_from_base_math_lv35_quicker_experiment_limr

PROMPT_TYPE="qwen25-math-cot"
MODEL_NAME_OR_PATH=$HDFS_HOME/checkpoints/$RUN_NAME
OUTPUT_DIR="Qwen2.5-Math-7B-Instruct-Math-Eval"

bash sh/eval_single_node.sh \
    --run_name ${RUN_NAME}  \
    --init_model_path $HDFS_HOME/model_hub/models--Qwen--Qwen2.5-Math-7B/ \
    --template ${PROMPT_TYPE}  \
    --tp_size 1