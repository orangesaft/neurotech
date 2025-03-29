import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
import xgboost as xgb


df = pd.read_csv("/Users/gwonjinlee/ds004302-download/participants.tsv", sep="\t")
print(df.head())

img = nib.load("/Users/gwonjinlee/ds004302-download/sub-01/anat/sub-01_T1w.nii.gz")
data = img.get_fdata()
print("Image shape:", data.shape)

plt.imshow(data[:, :, data.shape[2] // 2].T, cmap="gray", origin="lower")
plt.title("sub-01 (Mid Slice)")
plt.axis("off")
plt.show()

