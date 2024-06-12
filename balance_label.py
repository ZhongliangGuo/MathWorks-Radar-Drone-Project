import pandas as pd
from os.path import join

random_seed = 3407
label_path = r'/home/zg34/datasets/drone_project/all_label.csv'
save_path = r'/home/zg34/datasets/drone_project'
all_label = pd.read_csv(label_path)
drone = all_label.loc[all_label['drone_1_else_0'] == 1].sample(frac=1, random_state=random_seed).reset_index(drop=True)
non_drone = all_label.loc[all_label['drone_1_else_0'] == 0].sample(frac=1, random_state=random_seed).reset_index(
    drop=True)
num_data = min(len(drone), len(non_drone))
num_train = int(num_data * 0.9)

train_drone = drone.iloc[:num_train]
train_non_drone = non_drone.iloc[:num_train]
train_label = pd.concat([train_drone, train_non_drone],
                        ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)
train_label.to_csv(join(save_path, 'train_label.csv'), index=False)

eval_drone = drone.iloc[num_train:num_data]
eval_non_drone = non_drone.iloc[num_train:num_data]
eval_label = pd.concat([eval_drone, eval_non_drone],
                       ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)
eval_label.to_csv(join(save_path, 'eval_label.csv'), index=False)
