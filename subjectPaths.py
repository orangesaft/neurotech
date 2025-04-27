
for i in range(1, 78):
    synthsegPath = "/Users/gwonjinlee/Downloads/MRI_Data_For_Project/sub-" + str(i) + "/sub-" + str(i) + "_T1w_synthseg.nii.gz"
    print('"' +synthsegPath+ '",')

for i in range(1, 61):
    run_model = "run_model(" + str(i) + ")"
    print(run_model)





