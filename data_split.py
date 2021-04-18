
ID_patient = list(range(49,131)) # range of patient IDs
IDs_Hem = np.zeros((len(ID_patient))) # binary vector over patients (0 = no hemorage, 1 = hemorage)

import glob 
path =r'C:\YourFolder' #path to folder with .csv files

# iterate over patient
for i, id in enumerate(ID_patient):
    f = f"data/Patients_CT/{id:03d}/brain"

    if len(glob.glob(f+'/*_HGE_Seg.jpg')) > 0:
        IDs_Hem[i] = 1

IDs_hem_bool = np.array(IDs_hem, dtype=bool)
IDs_noHem = np.invert(IDs_hem_bool)

#IDs_noHem = np.ones((len(ID_patient))) - IDs_Hem

# split data
n_Hem = sum(IDs_Hem)
n_noHem = len(ID_patient)-n_Hem

IDs_Hem_patients = ID_patient[IDs_hem_bool]
IDs_noHem_patients = ID_patient[IDs_hem_bool]

split = [0.8,0.1,0.1]

# train (80)
n_train_Hem = n_Hem*split[0]

# test (10)

# validate (10)


    






