import nibabel as nib



df = pd.read_csv("/Users/gwonjinlee/ds004302-download/participants.tsv", sep="\t")
df['psyrats'] = df['psyrats'].fillna(0)
