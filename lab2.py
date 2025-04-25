import nibabel as nib
import numpy as np


# df = pd.read_csv("/Users/gwonjinlee/ds004302-download/participants.tsv", sep="\t")
# df['psyrats'] = df['psyrats'].fillna(0)

#to figure out synthseg numbers in the file
seg = nib.load("/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-01/sub-01_T1w_synthseg.nii.gz")
seg_data = seg.get_fdata()
unique_labels = np.unique(seg_data)
print(unique_labels)
#[ 0.  2.  3.  4.  5.  7.  8. 10. 11. 12. 13. 14. 15. 16. 17. 18. 24. 26.
# 28. 41. 42. 43. 44. 46. 47. 49. 50. 51. 52. 53. 54. 58. 60.]




