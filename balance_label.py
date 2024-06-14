import pandas as pd
from os.path import join

drone_candidates = [
    # 24GHz
    '2024-03-01_09-33-03_24GHz',  # DJI Mavic 3
    '2024-03-01_09-49-52_24GHz',  # DJI Mini 2
    '2024-03-01_10-14-20_24GHz',  # Autel Evo II
    '2024-03-01_10-46-33_24GHz',  # Yuneec H520E
    '2024-03-01_11-07-57_24GHz',  # DJI Matrice 210
    # 94GHz
    '2024-03-01_09-33-05_94GHz',  # DJI Mavic 3
    '2024-03-01_09-51-34_94GHz',  # DJI Mini 2
    '2024-03-01_10-14-24_94Ghz',  # Autel Evo II
    '2024-03-01_10-46-33_94GHz',  # Yuneec H520E
    '2024-03-01_11-07-57_94GHz',  # DJI Matrice 210
    # 207GHz
    '2024-03-01_09-33-03_207GHz',  # DJI Mavic 3
    '2024-03-01_09-49-45_207GHz',  # DJI Mini 2
    '2024-03-01_10-14-20_207GHz',  # Autel Evo II
    '2024-03-01_10-46-36_207GHz',  # Yuneec H520E
    '2024-03-01_12-45-57_207GHz',  # DJI Matrice 210
]

non_drone_candidates = [
    '2024-02-06_11-22-26_24GHz',  # 134
    '2024-02-06_12-31-07_24GHz',  # 177
    '2024-02-06_11-22-25_94GHz',  # 206
    '2024-02-06_12-31-00_94GHz',  # 275
    '2024-02-06_11-22-22_207GHz',  # 162
    '2024-02-06_12-30-58_207GHz',  # 275
]

random_seed = 3407
label_path = r'/home/zg34/datasets/drone_project/all_label.csv'
save_path = r'/home/zg34/datasets/drone_project'
all_label = pd.read_csv(label_path)
drone = all_label.loc[all_label['drone_1_else_0'] == 1].sample(frac=1, random_state=random_seed).reset_index(drop=True)
non_drone = all_label.loc[all_label['drone_1_else_0'] == 0].sample(frac=1, random_state=random_seed).reset_index(
    drop=True)


def subset_file_independent(full_label, candidates):
    eval_label_list = []
    for candidate in candidates:
        folder_name = candidate[:19]
        freq = candidate[20:]
        mask = (full_label['filename'].str.contains(folder_name)) & (full_label['radar_freq'] == freq)
        eval_label_list.append(full_label[mask])
    eval_label = pd.concat(eval_label_list, ignore_index=True).reset_index(drop=True)
    train_label = pd.merge(full_label, eval_label, how='left', indicator=True)
    train_label = train_label[train_label['_merge'] == 'left_only']
    train_label = (train_label.drop(columns=['_merge']))
    return train_label, eval_label


drone_train, drone_eval = subset_file_independent(drone, drone_candidates)
non_drone_train, non_drone_eval = subset_file_independent(non_drone, non_drone_candidates)

num_train = min(len(drone_train), len(non_drone_train))
train_drone = drone_train.iloc[:num_train]
train_non_drone = non_drone_train.iloc[:num_train]
train_label = pd.concat([train_drone, train_non_drone],
                        ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)
eval_label = pd.concat([drone_eval, non_drone_eval],
                       ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)
train_label.to_csv(join(save_path, 'train_label.csv'), index=False)
eval_label.to_csv(join(save_path, 'eval_label.csv'), index=False)
