import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def SAR_SYM_model(c=5, z=0.2, fet=1,  lu_map=None):
    """
    Species–Area Relationship (SAR) model.

    Parameters
    ----------
    file_path : str
        Path to the ASCII grid file (e.g., 'map.asc').
    c : float, optional
        Model constant (default = 5).
    z : float, optional
        Scaling exponent (default = 0.2).
    output_file : str, optional
        Output CSV file name (default = 'SAR_output.csv').

    Returns
    -------
    float
        Computed species richness (S).
        
    """

    #file_path="C:/Users/z3524121/Downloads/map.asc"
    
    #lu_file="input/10x10/land_use.asc"

    lu_map = np.asarray(lu_map)
    #print(lu_map)
    fert_file="input/10x10/soil_fertility.asc"
    
    output_file="SAR_output.csv"
    
    
    # --- Read the ASCII grid file ---
    '''lu_map = pd.read_csv(lu_file, 
                         delim_whitespace=True, 
                         header=None, 
                         skiprows=6)'''
    
    
    fert_map = pd.read_csv(fert_file, delim_whitespace=True, header=None, skiprows=6)

    # --- Determine area covered by forest (value = 7) ---
    forest_area = np.sum(lu_map == 7)

    # --- Calculate species richness (S = c * A^z) ---
    S = c * (forest_area ** z)

    # --- Replace NaN with 0 ---
    if np.isnan(S):
        S = 0
        
    # --- Identify arable land (0 < lu_map <= 5) ---
    arable_mask = (lu_map > 0) & (lu_map <= 5)

    # --- Calculate crop yield: log(lu * (1 + fertility)) ---
    yield_values = np.log(lu_map [arable_mask] * (1 + fet*fert_map.values[arable_mask]))

    # --- Replace NaN or invalid values with 0 ---
    yield_values = np.nan_to_num(yield_values, nan=0.0, neginf=0.0, posinf=0.0)

    # --- Sum up total yield ---
    yield_sum = np.sum(yield_values)

    # --- Write model output ---
    #pd.DataFrame([S]).to_csv(output_file, 
    #                         sep=";", 
    #                         index=False, 
    #                         header=False)

    #return S
    
    
    
    return {"S": S, "Yield": yield_sum}
