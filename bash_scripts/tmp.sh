accelerate launch \
  --main_process_port 29501 \
  scripts/train_wm.py \
  --dataset_root_path /home/ubuntu/irom-wm-ill/wm_data \
  --dataset_names v2_2025_12_12_1600 \
  --config droid_irom_finetune \
  --tag "1216-test-lambda"