from sklearn.model_selection import train_test_split
import glob
import numpy as np 
import os 
#from dataloader_test import IDs


def get_data_split_IDs(IDs, split = [0.8,0.1,0.1]):

    patient_IDs = []
    
    for ix in IDs:
        ID = ix.split('_')[1]
        patient_IDs.append(ID)
    
    patient_IDs = list(set(patient_IDs))
    
    ID_patient = np.array((patient_IDs),dtype=int)
    
    IDs_Hem = np.zeros((len(ID_patient))) # binary vector over patients (0 = no hemorage, 1 = hemorage)
    
    # iterate over patient
    for i, id in enumerate(ID_patient):
        f = f"data/Patients_CT/{id:03d}/brain"
    
        if len(glob.glob(f+'/*_HGE_Seg.jpg')) > 0:
            IDs_Hem[i] = 1
    
    #boolean arrays to find patients with hemorrhage 
    IDs_Hem_bool = np.array(IDs_Hem, dtype=bool)
    IDs_noHem_bool = np.invert(IDs_Hem_bool)
    
    
    # split data
    IDs_Hem_patients = ID_patient[IDs_Hem_bool]
    IDs_noHem_patients = ID_patient[IDs_noHem_bool]
    # print(IDs_Hem_patients)
    # print(len(IDs_Hem_patients))
    
    
    #the actual splitting 
    IDs_Hem_train, IDs_Hem_val_test = train_test_split(IDs_Hem_patients, test_size=split[1]+split[2], random_state = 11)
    IDs_noHem_train, IDs_noHem_val_test = train_test_split(IDs_noHem_patients, test_size=split[1]+split[2], random_state = 11)
    
    IDs_Hem_val, IDs_Hem_test = train_test_split(IDs_Hem_val_test, test_size=split[2] / (split[1]+split[2]), random_state = 11)
    IDs_noHem_val, IDs_noHem_test = train_test_split(IDs_noHem_val_test, test_size=split[2] / (split[1]+split[2]), random_state = 11)
    
    # print(IDs_noHem_train)
    # print(IDs_noHem_val)
    # print(IDs_noHem_test)
    
    
    train_IDs = []
    val_IDs = []
    test_IDs = []
    
    slices_total = []
    
    for pt in ID_patient:
        f = f"data/Patients_CT/{pt:03d}/bone"
        
        n_slices = len(os.listdir(f))
        
        slices_total.append(n_slices)
        
        for s in range(1,n_slices+1):
            if (pt in IDs_Hem_train) or (pt in IDs_noHem_train):
                train_IDs.append(f'pt_{pt:03d}_sl_{s}')
            elif (pt in IDs_Hem_val) or (pt in IDs_noHem_val): 
                val_IDs.append(f'pt_{pt:03d}_sl_{s}')
            else:
                test_IDs.append(f'pt_{pt:03d}_sl_{s}')

    return train_IDs, val_IDs, test_IDs
    

# a,b,c = get_data_split_IDs(IDs)

# print(b)
# print(len(b))

# train_IDs, val_IDs, test_IDs = get_data_split_IDs(IDs)



