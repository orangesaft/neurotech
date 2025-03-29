import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
import os

df = pd.read_csv("/Users/gwonjinlee/ds004302-download/participants.tsv", sep="\t")
print(df.head())


img = nib.load("/Users/gwonjinlee/ds004302-download/sub-01/anat/sub-01_T1w.nii.gz")
data = img.get_fdata()
print("Image shape:", data.shape)

plt.imshow(data[:, :, data.shape[2] // 2].T, cmap="gray", origin="lower")
plt.title("sub-01 (Mid Slice)")
plt.axis("off")
plt.show()


df['psyrats'] = df['psyrats'].fillna(0)

x = []
y = []

for _, row in df.iterrows():
    subject_id = row['participant_id']
    psyrats_score = row['psyrats']

    try:
        # Path to the subject's MRI
        path = f"/Users/gwonjinlee/ds004302-download/{subject_id}/anat/{subject_id}_T1w.nii.gz"

        # Load and extract middle slice
        img = nib.load(path)
        data = img.get_fdata()
        mid_slice = data[:, :, data.shape[2] // 2]
        flat_slice = mid_slice.flatten()

        # Append features and label
        x.append(flat_slice)
        y.append(psyrats_score)

    except Exception as e:
        print(f"Skipped {subject_id}: {e}")

numpyX = np.array(x)
numpyY = np.array(y)

print("numpyX shape:", numpyX.shape)
print("numpyY shape:", numpyY.shape)