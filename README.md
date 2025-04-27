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

# Results
## lab.py
![img.png](img.png)
## lab2.py
Label of Interest: 2
Mean Squared Error: 241.1924280089747
R² Score: -0.6532107567787522

Label of Interest: 3
Mean Squared Error: 410.03222858863955
R² Score: -1.8104932502420006

Label of Interest: 4
Mean Squared Error: 390.68634569289935
R² Score: -1.6778903241607983

Label of Interest: 5
Mean Squared Error: 443.51038202780205
R² Score: -2.0399633204245258

Label of Interest: 7
Mean Squared Error: 497.72852473260656
R² Score: -2.4115919717552092

Label of Interest: 8
Mean Squared Error: 274.72934698447324
R² Score: -0.8830836249164224

Label of Interest: 10
Mean Squared Error: 268.5551591076265
R² Score: -0.8407637482244552

Label of Interest: 11
Mean Squared Error: 195.77940874662593
R² Score: -0.341935263754062

Label of Interest: 12
Mean Squared Error: 251.73973075612994
R² Score: -0.7255053744022801

Label of Interest: 13
Mean Squared Error: 397.1183774038849
R² Score: -1.721977545722114

Label of Interest: 14
Mean Squared Error: 335.8384609193375
R² Score: -1.3019452174145782

Label of Interest: 15
Mean Squared Error: 348.57972608008356
R² Score: -1.3892779616163655

Label of Interest: 16
Mean Squared Error: 346.35673803095716
R² Score: -1.3740408839628762

Label of Interest: 17
Mean Squared Error: 296.28940411388345
R² Score: -1.030863215914939

Label of Interest: 18
Mean Squared Error: 156.91084235418558
R² Score: -0.07551756320269787

Label of Interest: 24
Mean Squared Error: 340.79731625791726
R² Score: -1.3359348125885395

Label of Interest: 26
Mean Squared Error: 373.32809849042843
R² Score: -1.5589112947159691

Label of Interest: 28
Mean Squared Error: 357.943999486925
R² Score: -1.4534637142679014

Label of Interest: 41
Mean Squared Error: 207.84527415382306
R² Score: -0.42463860003077425

Label of Interest: 42
Mean Squared Error: 284.0239680385215
R² Score: -0.946791957858629

Label of Interest: 43
Mean Squared Error: 373.0715473369216
R² Score: -1.5571528102969405

Label of Interest: 44
Mean Squared Error: 451.0047461402848
R² Score: -2.0913321111790686

Label of Interest: 46
Mean Squared Error: 245.1857940474152
R² Score: -0.6805825766364597

Label of Interest: 47
Mean Squared Error: 519.4501934334801
R² Score: -2.560479300631604

Label of Interest: 49
Mean Squared Error: 269.7872164936758
R² Score: -0.8492086672478238

Label of Interest: 50
Mean Squared Error: 254.08301750213505
R² Score: -0.7415670181557423

Label of Interest: 51
Mean Squared Error: 370.9384752183369
R² Score: -1.542532045455609

Label of Interest: 52
Mean Squared Error: 385.9074321151954
R² Score: -1.6451341078998043

Label of Interest: 53
Mean Squared Error: 204.948989283955
R² Score: -0.40478652863248277

Label of Interest: 54
Mean Squared Error: 247.33523911478218
R² Score: -0.6953155669538171

Label of Interest: 58
Mean Squared Error: 231.76053207826934
R² Score: -0.5885614975205815

Label of Interest: 60
Mean Squared Error: 312.8558090150058
R² Score: -1.1444147026252458


