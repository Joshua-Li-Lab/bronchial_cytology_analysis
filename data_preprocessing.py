import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(input_file):
    # Read data from excel
    df = pd.read_excel(input_file)
    
    # Calculate age in years
    df['Age'] = (df['Admission Date (yyyy-mm-dd)'].dt.year - df['Date of Birth (yyyy-mm-dd)'].dt.year) - \
                ((df['Admission Date (yyyy-mm-dd)'].dt.month < df['Date of Birth (yyyy-mm-dd)'].dt.month) | 
                 ((df['Admission Date (yyyy-mm-dd)'].dt.month == df['Date of Birth (yyyy-mm-dd)'].dt.month) & 
                  (df['Admission Date (yyyy-mm-dd)'].dt.day < df['Date of Birth (yyyy-mm-dd)'].dt.day)))
    
    # Create the AGE55ormore column
    df['AGE55ormore'] = (df['Age'] >= 55).astype(int)
    
    # Replacements
    male_female = {'M': 1, 'F': 0}
    high_normal_low = {'H': 3, np.nan: 2, 'L': 1}
    
    high_normal_low_columns = [
        'APTT_Flagging', 'Albumin_Flagging', 'Basophil, absolute_Flagging', 'C-Reactive Protein_Flagging',
        'Creatinine_Flagging', 'Eosinophil, absolute_Flagging',
        'Haemoglobin, Blood_Flagging', 'Lactate Dehydrogenase_Flagging', 'Lymphocyte, absolute_Flagging',
        'MCH_Flagging', 'MCHC_Flagging', 'MCV_Flagging',
        'Neutrophil, absolute_Flagging', 'Platelet_Flagging',
        'Protein, Total_Flagging', 'Prothrombin Time_Flagging', 'WBC_Flagging'
    ]
    
    df[high_normal_low_columns] = df[high_normal_low_columns].replace(high_normal_low)
    df[['Sex']] = df[['Sex']].replace(male_female)
    
    return df

def split_and_export_data(df):
    train_df, val_df = train_test_split(df, test_size=0.5, random_state=42)
    return train_df, val_df





