num_gpus=3

deepspeed --num_gpus $num_gpus trainer.py \
    --deepspeed ./gpu.json \
    --output_dir ./output/Qwen \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --num_train_epochs 10 \
    --save_steps 100 \
    --learning_rate 1e-4 \
    --bf16 True \
    --save_on_each_node False \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "linear"
