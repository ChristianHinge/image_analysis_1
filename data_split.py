from sklearn.model_selection import train_test_split
import glob
import numpy as np 
import os 

patient_id_folder = "C:/Users/simon/OneDrive/Dokumenter/Adv_image_analysis/mini-project/image_analysis_1/data/Patients_CT"

def get_data_split_IDs(patient_id_folder, split = [0.8,0.1,0.1]):

    ID_patient = []
    
    ids_path = glob.glob(patient_id_folder + '/*')
    
    for ids in ids_path:
        #print(ids)
        ID = ids.split("\\")
        ID_patient.append(ID[-1])
    
    ID_patient = np.array((ID_patient),dtype=int)
    
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
    

    #the actual splitting 
    IDs_Hem_train, IDs_Hem_val_test = train_test_split(IDs_Hem_patients, test_size=split[1]+split[2])
    IDs_noHem_train, IDs_noHem_val_test = train_test_split(IDs_noHem_patients, test_size=split[1]+split[2])
    
    IDs_Hem_val, IDs_Hem_test = train_test_split(IDs_Hem_val_test, test_size=split[2] / (split[1]+split[2]))
    IDs_noHem_val, IDs_noHem_test = train_test_split(IDs_noHem_val_test, test_size=split[2] / (split[1]+split[2]))
    
    
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
                train_IDs.append(f'pt_{pt}_sl_{s}')
            elif (pt in IDs_Hem_val) or (pt in IDs_noHem_val): 
                val_IDs.append(f'pt_{pt}_sl_{s}')
            else:
                test_IDs.append(f'pt_{pt}_sl_{s}')
    
    return train_IDs, val_IDs, test_IDs


# train_IDs, val_IDs, test_IDs = get_data_split_IDs(patient_id_folder)

# print(sum(slices_total))
# print(len(train_IDs)+len(val_IDs)+len(test_IDs))








