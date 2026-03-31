import numpy as np
def candidate_to_policy(candidate):
    # candidate will be x1_map / x2_map (numpy array) passed on for model evaluation
    return {"lu_map": np.asarray(candidate)}
