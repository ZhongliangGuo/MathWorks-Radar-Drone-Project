import os
from os.path import join
from constant import IMPLEMENTED_NETS

bash_save_path = r'./bash/arch'
os.makedirs(bash_save_path, exist_ok=True)
py_path = '/workspace/DroneClassificationNet/train.py'
data_root = '/workspace/datasets/drone_project/data'
train_label_path = '/workspace/datasets/drone_project/train_label.csv'
eval_label_path = '/workspace/datasets/drone_project/eval_label.csv'
output_dir = './logs'
epochs = 100
batch_size = 64
lr = 1e-5
eps = 1e-5
random_seed = 3407
ckpt_interval = 20
num_workers = 8
bash_list = []
for arch in IMPLEMENTED_NETS:
    template = (
        f"python {py_path} \\\n"
        f"  --task {'binary'} \\\n"
        f"  --arch {arch} \\\n"
        f"  --use_pretrained \\\n"
        f"  --epochs {epochs} \\\n"
        f"  --batch_size {batch_size} \\\n"
        f"  --lr {lr} \\\n"
        f"  --eps {eps} \\\n"
        f"  --ckpt_interval {ckpt_interval} \\\n"
        f"  --num_workers {num_workers} \\\n"
        f"  --random_seed {random_seed} \\\n"
        f"  --data_root {data_root} \\\n"
        f"  --train_label_path {train_label_path} \\\n"
        f"  --eval_label_path {eval_label_path} \\\n"
        f"  --output_dir {output_dir}"
    )
    with open(join(bash_save_path, f'{arch}.sh'), mode='w+') as f:
        print(template, file=f)
    bash_list.append(f'{arch}.sh')
with open(join(bash_save_path, f'run_me.sh'), mode='w+') as f:
    print('\n'.join(f"bash {sub}" for sub in bash_list), file=f)
