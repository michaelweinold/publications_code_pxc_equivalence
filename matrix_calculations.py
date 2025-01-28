# %%

import gzip
import pickle
import time
import numpy as np
import pandas as pd

path = '/Users/michaelweinold/Library/CloudStorage/OneDrive-TheWeinoldFamily/Documents/University/PhD/Data/HLCA Matrices/hybrid_system.pickle'
with gzip.open(path, 'rb') as pickle_file:
    picklefile = pickle.load(file=pickle_file)

A_P = picklefile['A_ff'].todense()
A_S = picklefile['A_io'].todense()
C_U = picklefile['A_io_f'].todense()
A_H = np.block(
    [
        [np.eye(A_P.shape[0]) - A_P, np.zeros((A_P.shape[0], A_S.shape[0]))],
        [C_U, np.eye(A_S.shape[0]) - A_S]
    ]
)

# %%

def generate_final_demand_vector(
    number_of_sectors: int,
    sector_index: int,
    demand_amount: float
) -> np.ndarray:
    f_vector = np.zeros(number_of_sectors)
    f_vector[sector_index] = demand_amount
    return f_vector

f_vector_H = generate_final_demand_vector(
    number_of_sectors=A_H.shape[0],
    sector_index=1,
    demand_amount=1
)

list_computation_times = []
list_results = []
for i in range(5):
    start = time.time()
    result = np.linalg.solve(A_H, f_vector_H)
    end = time.time()
    list_computation_times.append((end - start)/60)
    list_results.append(result)




# %%

isic_list = picklefile['PRO_f']['ISIC']
isic_list_first_two = isic_two = [str(string)[:2] for string in isic_list]

isic_divisions = {
    'A': ['01', '02', '03'],
    'B': ['05', '06', '07', '08', '09'],
    'C': [
        '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
        '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', 
        '31', '32', '33'
    ],
    'D': ['35'],
    'E': ['36', '37', '38', '39'],
    'F': ['41', '42', '43'],
    'G': ['45', '46', '47'],
    'H': ['49', '50', '51', '52', '53'],
    'I': ['55', '56'],
    'J': ['58', '59', '60', '61', '62', '63'],
    'K': ['64', '65', '66'],
    'L': ['68'],
    'M': ['69', '70', '71', '72', '73', '74', '75'],
    'N': ['77', '78', '79', '80', '81', '82'],
    'O': ['84'],
    'P': ['85'],
    'Q': ['86', '87', '88'],
    'R': ['90', '91', '92', '93'],
    'S': ['94', '95', '96'],
    'T': ['97', '98'],
    'U': ['99']
}

def find_indices(arr, value):
    return [index for index, element in enumerate(arr) if element == value]

dict_isic_indices = {}

for isic_letter in isic_divisions.keys():
    list_of_indices = []
    for isic_code in isic_divisions[isic_letter]:
        list_of_indices += find_indices(isic_two, isic_code)
    dict_isic_indices[isic_letter] = list_of_indices

def sample_random_indices(list, number_of_indices):
    if len(list) < number_of_indices:
        number_of_indices = len(list)
    return np.random.choice(list, number_of_indices, replace=False)

dict_random_indices = {
    isic_division: sample_random_indices(indices, 20) for isic_division, indices in dict_isic_indices.items() if indices != []
}

def generate_final_demand_vector(
    number_of_sectors: int,
    sector_index: int,
    demand_amount: float
) -> np.ndarray:
    f_vector = np.zeros(number_of_sectors)
    f_vector[sector_index] = demand_amount
    return f_vector


f_vector_H = generate_final_demand_vector(
    number_of_sectors=A_H.shape[0],
    sector_index=1,
    demand_amount=1
)

dict_result_dataframes = {}

for isic_section in dict_random_indices.keys():
    print('Now processing ISIC section:', isic_section)
    computation_times = []
    for sector_index in dict_random_indices[isic_section]:
        print('Index of sector:', sector_index)
        f_vector = generate_final_demand_vector(
            number_of_sectors=A_H.shape[0],
            sector_index=sector_index,
            demand_amount=1
        )
        start = time.time()
        np.linalg.solve(A_H, f_vector)
        end = time.time()
        computation_times.append((end - start)/60)

    df_computation_times = pd.DataFrame(
        {
            'sector_index': dict_random_indices[isic_section],
            'computation_times': computation_times
        }
    )

    dict_result_dataframes[isic_section] = df_computation_times

import pickle
with open('results_computation_times.pkl', 'wb') as f:
    pickle.dump(dict_result_dataframes, f)
# %%
