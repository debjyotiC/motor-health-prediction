import os
from os import listdir
from os.path import join
import pandas as pd

dataset = "dataset/raw/train"

data_frames = []

for target, folder in enumerate(listdir(dataset)):
    print(folder)
    for file in listdir(join(dataset, folder)):
        file_path = join(dataset, folder, file)

        df = pd.read_csv(file_path)

        df = df.drop(columns=['class_label'], errors='ignore')

        ax = df['ax']
        ay = df['ay']
        az = df['az']

        gx = df['gx']
        gy = df['gy']
        gz = df['gz']

        df_selected = df[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].copy()
        df_selected['Label'] = target

        data_frames.append(df_selected)

converted_df = pd.concat(data_frames, ignore_index=True)

output_path = f"dataset/digested/{dataset.split("/")[2]}.csv"

os.makedirs(os.path.dirname(output_path), exist_ok=True)
converted_df.to_csv(output_path, index=False)

print(f"Saved: {output_path}")