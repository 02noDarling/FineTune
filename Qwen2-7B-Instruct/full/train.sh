num_gpus=3

deepspeed --num_gpus $num_gpus trainer.py \
    --deepspeed ./gpu.json \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --logging_steps=10 \
    --num_train_epochs=1 \
    --save_steps=100 \
    --learning_rate=1e-4 \
    --bf16=True \
    --save_on_each_node=False 
