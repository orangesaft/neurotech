import os
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#reading the tsv file, .fillna(0) because HC should have psyrats = 0.
df = pd.read_csv("/Users/gwonjinlee/ds004302-download/participants.tsv", sep="\t")
df['psyrats'] = df['psyrats'].fillna(0)

#feature and label
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
    numpyX, numpyY, test_size=0.5, random_state=42
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
    num_boost_round=200,
    evals=evals,
    early_stopping_rounds=20,
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


# ###     HC vs AVH
# ###     HC vs AVH
# ###     HC vs AVH
# ###     HC vs AVH
#
#
# import os
# import numpy as np
# import pandas as pd
# import nibabel as nib
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
#
# # Step 1: Load and label participants
# df2 = pd.read_csv("/Users/gwonjinlee/ds004302-download/participants.tsv", sep="\t")
# # df2['group'] = df2['psyrats'].replace({'AVH-': 'AVH', 'AVH+': 'AVH'}).fillna(df2['group'])
# df2['label'] = df2['group'].apply(lambda x: 1 if x == 'AVH-' or x == 'AVH+' else 0)  # 1 = AVH, 0 = HC
#
# # Step 2: Extract MRI mid-slices
# x_cls = []
# y_cls = []
#
# for _, row in df2.iterrows():
#     subject_id = row['participant_id']
#     label = row['label']
#
#     try:
#         path = f"/Users/gwonjinlee/ds004302-download/{subject_id}/anat/{subject_id}_T1w.nii.gz"
#         img = nib.load(path)
#         data = img.get_fdata()
#         mid_slice = data[:, :, data.shape[2] // 2]
#         flat_slice = mid_slice.flatten()
#
#         x_cls.append(flat_slice)
#         y_cls.append(label)
#         print(f"{subject_id} → label: {label}")  # Debug: Show what's being loaded
#
#     except Exception as e:
#         print(f"Skipped {subject_id}: {e}")
#
# # Step 3: Convert to NumPy and filter valid labels
# X_cls = np.array(x_cls)
# Y_cls = np.array(y_cls)
#
# # Filter: Only keep 0 (HC) and 1 (AVH)
# mask = (Y_cls == 0) | (Y_cls == 1)
# X_cls = X_cls[mask]
# Y_cls = Y_cls[mask]
#
# # Check if we still have both labels
# unique, counts = np.unique(Y_cls, return_counts=True)
# print("Filtered label distribution:", dict(zip(unique, counts)))
#
# # Step 4: Stratified train/test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X_cls, Y_cls, test_size=0.5, random_state=42, stratify=Y_cls
# )
#
# # Step 5: Convert labels to float (for XGBoost)
# y_train = y_train.astype(float)
# y_test = y_test.astype(float)
#
# # Step 6: Train with XGBoost
# dtrain = xgb.DMatrix(X_train, label=y_train)
# dtest = xgb.DMatrix(X_test, label=y_test)
#
# params_cls = {
#     'objective': 'binary:logistic',
#     'max_depth': 2,
#     'eta': 0.1,
#     'subsample': 0.8,
#     'colsample_bytree': 0.8,
#     'eval_metric': 'logloss',
#     'seed': 42
# }
#
# model_cls = xgb.train(
#     params_cls,
#     dtrain,
#     num_boost_round=200,
#     evals=[(dtrain, 'train'), (dtest, 'test')],
#     early_stopping_rounds=20,
#     verbose_eval=True
# )
#
# # Step 7: Predict and evaluate
# y_pred_proba = model_cls.predict(dtest)
# y_pred = (y_pred_proba > 0.5).astype(int)
#
# print(classification_report(y_test, y_pred))
