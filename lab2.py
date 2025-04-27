import nibabel as nib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("/Users/gwonjinlee/ds004302-download/participants.tsv", sep="\t")
df['psyrats'] = df['psyrats'].fillna(0)

#label = psyrats score
Y = df['psyrats'].values
#print(Y)

#to figure out synthseg numbers in the file
# seg = nib.load("/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-01/sub-01_T1w_synthseg.nii.gz")
# seg_data = seg.get_fdata()
# unique_labels = np.unique(seg_data)
# print(unique_labels)
#[ 0.  2.  3.  4.  5.  7.  8. 10. 11. 12. 13. 14. 15. 16. 17. 18. 24. 26.
# 28. 41. 42. 43. 44. 46. 47. 49. 50. 51. 52. 53. 54. 58. 60.]

#labels    structures
# 0         background
# 2         left cerebral white matter
# 3         left cerebral cortex
# 4         left lateral ventricle
# 5         left inferior lateral ventricle
# 7         left cerebellum white matter
# 8         left cerebellum cortex
# 10        left thalamus
# 11        left caudate
# 12        left putamen
# 13        left pallidum
# 14        3rd ventricle
# 15        4th ventricle
# 16        brain-stem
# 17        left hippocampus
# 18        left amygdala
# 26        left accumbens area
# 24        CSF
# 28        left ventral DC
# 41        right cerebral white matter
# 42        right cerebral cortex
# 43        right lateral ventricle
# 44        right inferior lateral ventricle
# 46        right cerebellum white matter
# 47        right cerebellum cortex
# 49        right thalamus
# 50        right caudate
# 51        right putamen
# 52        right pallidum
# 53        right hippocampus
# 54        right amygdala
# 58        right accumbens area
# 60        right ventral DC

#Function to count voxels for a specific region
def extract_voxel_count_for_label(seg_path, label):
    seg = nib.load(seg_path)
    seg_data = seg.get_fdata().astype(np.int32)
    count = np.sum(seg_data == label)
    return count

#Paths
subject_paths = [
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-01/sub-01_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-02/sub-02_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-03/sub-03_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-04/sub-04_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-05/sub-05_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-06/sub-06_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-07/sub-07_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-08/sub-08_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-09/sub-09_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-10/sub-10_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-11/sub-11_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-12/sub-12_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-13/sub-13_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-14/sub-14_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-15/sub-15_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-16/sub-16_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-17/sub-17_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-18/sub-18_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-19/sub-19_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-20/sub-20_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-21/sub-21_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-22/sub-22_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-23/sub-23_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-24/sub-24_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-25/sub-25_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-26/sub-26_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-27/sub-27_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-28/sub-28_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-29/sub-29_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-30/sub-30_T1w_synthseg.nii.gz",
    # "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-31/sub-31_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-32/sub-32_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-33/sub-33_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-34/sub-34_T1w_synthseg.nii.gz",
    # "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-35/sub-35_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-36/sub-36_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-37/sub-37_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-38/sub-38_T1w_synthseg.nii.gz",
    # "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-39/sub-39_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-40/sub-40_T1w_synthseg.nii.gz",
    # "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-41/sub-41_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-42/sub-42_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-43/sub-43_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-44/sub-44_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-45/sub-45_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-46/sub-46_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-47/sub-47_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-48/sub-48_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-49/sub-49_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-50/sub-50_T1w_synthseg.nii.gz",
    # "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-51/sub-51_T1w_synthseg.nii.gz",
    # "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-52/sub-52_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-53/sub-53_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-54/sub-54_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-55/sub-55_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-56/sub-56_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-57/sub-57_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-58/sub-58_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-59/sub-59_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-60/sub-60_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-61/sub-61_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-62/sub-62_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-63/sub-63_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-64/sub-64_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-65/sub-65_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-66/sub-66_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-67/sub-67_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-68/sub-68_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-69/sub-69_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-70/sub-70_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-71/sub-71_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-72/sub-72_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-73/sub-73_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-74/sub-74_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-75/sub-75_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-76/sub-76_T1w_synthseg.nii.gz",
    "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-77/sub-77_T1w_synthseg.nii.gz"
]


label_of_interest = 2

# Extract voxel counts
X = []
for seg_path in subject_paths:
    voxel_count = extract_voxel_count_for_label(seg_path, label_of_interest)
    X.append([voxel_count])

X = np.array(X)

#print(X.shape)
#print(X)

#Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Train XGBoost
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

#Predict and evaluate
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))




