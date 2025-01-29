# %%

import gzip
import pickle
import time
import numpy as np
import pandas as pd
import pickle

path = '/Users/michaelweinold/Library/CloudStorage/OneDrive-TheWeinoldFamily/Documents/University/PhD/Data/HLCA Matrices/hybrid_system.pickle'
with gzip.open(path, 'rb') as pickle_file:
    picklefile = pickle.load(file=pickle_file)

# See also:
# https://github.com/MaximeAgez/pylcaio/blob/505898a39144ebc53c109e485644e3ea055ae0ae/src/pylcaio.py#L46

A_P = picklefile['A_ff'].todense().A
A_S = picklefile['A_io'].todense().A
C_U = picklefile['A_io_f'].todense().A
A_H = np.block(
    [
        [np.eye(A_P.shape[0]) - A_P, np.zeros((A_P.shape[0], A_S.shape[0]))],
        [C_U, np.eye(A_S.shape[0]) - A_S]
    ]
)

B_S = picklefile['F_io'].todense().A
B_P = picklefile['F_f'].todense().A
B_H = np.block(
    [
        [B_P, np.zeros((B_P.shape[0], B_S.shape[1]))],
        [np.zeros((B_S.shape[0], B_P.shape[1])), B_S]
    ]
)

C_P_climate = picklefile['C_f'].todense().A[0,:]
C_S_climate = picklefile['C_io'].todense().A[0,:]
C_H_climate = np.concatenate((C_P_climate, C_S_climate), axis=0).T


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
    isic_division: sample_random_indices(indices, 10) for isic_division, indices in dict_isic_indices.items() if indices != []
}


def generate_final_demand_vector(
    number_of_sectors: int,
    sector_index: int,
    demand_amount: float
) -> np.ndarray:
    f_vector = np.zeros(number_of_sectors)
    f_vector[sector_index] = demand_amount
    return f_vector


def compute_environmental_burden(
    A_H: np.ndarray,
    B_H: np.ndarray,
    C_H_climate: np.ndarray,
    sector_index: int,
) -> tuple[float, float]:
    f_vector_H = generate_final_demand_vector(
        number_of_sectors=A_H.shape[0],
        sector_index=sector_index,
        demand_amount=1
    )
    start = time.time()
    vec_intermediate_demand = np.linalg.solve(A_H, f_vector_H)
    vec_environmental_flows = np.dot(B_H, vec_intermediate_demand)
    scal_environmental_burden = np.dot(C_H_climate, vec_environmental_flows)
    end = time.time()
    computation_time = end - start
    return scal_environmental_burden, computation_time


for isic_section in dict_random_indices.keys():
    print('Now processing ISIC section:', isic_section)
    list_of_list_computation_times = []
    list_of_list_results = []
    list_cv = []
    for sector_index in dict_random_indices[isic_section]:
        list_results = []
        list_computation_times = []
        print('Index of sector:', sector_index)
        for _ in range(0, 4):
            result = compute_environmental_burden(
                A_H=A_H,
                B_H=B_H,
                C_H_climate=C_H_climate,
                sector_index=sector_index
            )
            list_results.append(result[0])
            print(f'Computing environmental burden took {result[1]} seconds.')
            list_computation_times.append(result[1])
        
        std_result = np.std(list_results, ddof=1)
        mean_result = np.mean(list_results)
        list_cv.append(std_result / mean_result)
        list_of_list_results.append(list_results)
        list_of_list_computation_times.append(list_computation_times)

    df_results_isic_section = pd.DataFrame(
        {
            'sector_index': dict_random_indices[isic_section],
            'computation_times': list_of_list_computation_times,
            'environmental_burden': list_of_list_results,
            'cv': list_cv
        }
    )
    with open(f'results_section_{isic_section}.pkl', 'wb') as f:
        pickle.dump(df_results_isic_section, f)
