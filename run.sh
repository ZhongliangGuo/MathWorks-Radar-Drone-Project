python train.py \
  --arch resnet18 \
  --epochs 100 \
  --batch_size 64 \
  --lr 1e-5 \
  --random_seed 3407 \
  --train_label_path /home/zg34/datasets/drone_project/train_label.csv \
  --eval_label_path /home/zg34/datasets/drone_project/eval_label.csv \
  --data_root /home/zg34/datasets/drone_project/data