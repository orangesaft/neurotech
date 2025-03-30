import os
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



df = pd.read_csv("/Users/gwonjinlee/ds004302-download/participants.tsv", sep="\t")
# #print(df.head())
#
#
# img = nib.load("/Users/gwonjinlee/ds004302-download/sub-01/anat/sub-01_T1w.nii.gz")
# data = img.get_fdata()
# #print("Image shape:", data.shape)
#
# plt.imshow(data[:, :, data.shape[2] // 2].T, cmap="gray", origin="lower")
# plt.title("sub-01 (Mid Slice)")
# plt.axis("off")
# #plt.show()


df['psyrats'] = df['psyrats'].fillna(0)

x = []
y = []

for _, row in df.iterrows():
    subject_id = row['participant_id']
    psyrats_score = row['psyrats']

    try:
        path = f"/Users/gwonjinlee/ds004302-download/{subject_id}/anat/{subject_id}_T1w.nii.gz"

        img = nib.load(path)
        data = img.get_fdata()
        mid_slice = data[:, :, data.shape[2] // 2]
        flat_slice = mid_slice.flatten()

        x.append(flat_slice)
        y.append(psyrats_score)

    except Exception as e:
        print(f"Skipped {subject_id}: {e}")

numpyX = np.array(x)
numpyY = np.array(y)

print("numpyX shape:", numpyX.shape)
print("numpyY shape:", numpyY.shape)

numpyX_train, numpyX_test, numpyY_train, numpyY_test = train_test_split(
    numpyX, numpyY, test_size=0.3, random_state=42
)

dtrain = xgb.DMatrix(numpyX_train, label=numpyY_train)
dtest = xgb.DMatrix(numpyX_test, label=numpyY_test)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 2,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 1.0,
    'alpha': 0.5,
    'eval_metric': 'rmse',
    'seed': 42
}

evals = [(dtrain, 'train'), (dtest, 'test')]

model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=evals,
    early_stopping_rounds=10,
    verbose_eval=True
)

numpyY_pred = model.predict(dtest)

mse = mean_squared_error(numpyY_test, numpyY_pred)
r2 = r2_score(numpyY_test, numpyY_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)

plt.scatter(numpyY_test, numpyY_pred, alpha=0.7)
plt.xlabel("Actual PSYRATS")
plt.ylabel("Predicted PSYRATS")
plt.title("Actual vs Predicted PSYRATS Scores")
plt.plot([min(numpyY_test), max(numpyY_test)], [min(numpyY_test), max(numpyY_test)], color='red')  # ideal line
plt.grid(True)
plt.show()




