# neurotech
- https://openneuro.org/datasets/ds004302/versions/1.0.1


- https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0276975#sec008


- https://www.cambridge.org/core/journals/psychological-medicine/article/scales-to-measure-dimensions-of-hallucinations-and-delusions-the-psychotic-symptom-rating-scales-psyrats/F98A9A5A0D5CB9715161C1547DB010B8


- https://github.com/BBillot/SynthSeg/blob/master/data/labels%20table.txt

### topic

- neuroimaging data (MRI) to figure out or predict structural change in the brains of schizophrenia patients experiencing AVH (auditory verbal hallucination)

### method (lab.py)

- 71 brain scans (HC, AVH-, AVH+)
- convert MRI scan data (3D) to 1D data to be used for machine learning with XGBoost
- compare MRI scans with PSYRATS scores (represents AVH)

### method (lab2.py)
- Synthseg
- Voxels


### what's different about this
- didn't use FSL, a common software for neuroimaing data analysis. (researchers in the original paper used FSL)
- used XGBoost, not a common choice for machine learning with images.