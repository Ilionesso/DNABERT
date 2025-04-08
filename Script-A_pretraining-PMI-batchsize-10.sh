export KMER=6
export TRAIN_FILE=data/dnabert_original_train_pretraining_data.txt
export TEST_FILE=data/dnabert_original_valid_pretraining_data.txt
export SOURCE=/home.nfs/sheikili/DNABERT/
export OUTPUT_PATH=output$KMER

#    --eval_all_checkpoints \

python examples/run_pretrain_pmi.py \
    --output_dir $OUTPUT_PATH \
    --model_type=dna \
    --tokenizer_name=dna$KMER \
    --config_name=$SOURCE/src/transformers/dnabert-config/bert-config-$KMER/config.json \
    --train_data_file=$TRAIN_FILE \
    --do_train \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --gradient_accumulation_steps 25 \
    --per_gpu_train_batch_size 10 \
    --per_gpu_eval_batch_size 6 \
    --save_steps 800 \
    --save_total_limit 20 \
    --max_steps 10000 \
    --evaluate_during_training \
    --logging_steps 800 \
    --line_by_line \
    --learning_rate 8e-4 \
    --block_size 512 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.01 \
    --beta1 0.9 \
    --beta2 0.98 \
    --warmup_steps 500 \
    --n_process 24 \
    --overwrite_output_dir \
#    --should_continue
#    --mlm_probability 0.0176 \
