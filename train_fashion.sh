accelerate launch --num_processes 8 --mixed_precision "fp16" \
  tutorial_train.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
  --image_encoder_path="h94/IP-Adapter" \
  --data_json_file="/workspace/Fashion_image/data.json" \
  --data_root_path="/workspace/Fashion_image/train/" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=16 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="fashion_image" \
  --save_steps=10000