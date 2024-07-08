accelerate launch --num_processes 8 --mixed_precision "fp16" \
  market_train.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
  --image_encoder_path="h94/IP-Adapter" \
  --data_json_file="/workspace/Market-1501/train_data.json" \
  --data_root_path="/workspace/Market-1501/bounding_box_train" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=16 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="fashion_image" \
  --save_steps=10000 \
