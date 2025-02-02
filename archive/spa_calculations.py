# %%

import pandas as pd
import pyspa
import time
from multiprocessing import Pool

df_infosheet = pd.read_csv(
    filepath_or_buffer='../Infosheet_template.csv',
    header=0,    
)
df_sectors: pd.DataFrame = df_infosheet[['Sector number', 'Name']]

list_results_dataframes = []
list_cutoff = [0.1, 0.01, 0.001, 0.0001, 0.00001]

def process_sector(sector_id):
    list_comp_time = []
    list_spa_coverage = []
    for cutoff in list_cutoff:
        start_time = time.time()
        sc = pyspa.get_spa(
            target_ID=sector_id,
            max_stage=20,
            a_matrix='../A_matrix_template.csv',
            infosheet='../Infosheet_template.csv',
            thresholds={'GHG_emissions': cutoff},
            thresholds_as_percentages=True,
            zero_indexing=True,
        )
        end_time = time.time()
        list_comp_time.append(end_time - start_time)
        list_spa_coverage.append(sc.get_coverage_of('GHG_emissions'))
    df_results = pd.DataFrame(
        data={
            'Cutoff': list_cutoff,
            'Computation time': list_comp_time,
            'SPA coverage': list_spa_coverage,
        }
    )
    return df_results

# %%

if __name__ == '__main__':
    with Pool() as pool:
        list_results_dataframes = pool.map(process_sector, df_sectors.index)

import pickle
with open('results_dataframes.pkl', 'wb') as f:
    pickle.dump(list_results_dataframes, f)

# %%

import matplotlib.pyplot as plt

