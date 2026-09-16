
DOMAIN_CONFIG = {
    "Countries": {
        "stim":  "../Data/Behavioral Data/design_data_countries.csv",
        "data":  "../Data/Behavioral Data/data_analysis_countries.csv",
        "n_dim": 10,
        "ub": 100,
    },
    "Mammals": {
        "stim": "../Data/Behavioral Data/design_data_mammals.csv",
        "data": "../Data/Behavioral Data/data_analysis_mammals.csv",
        "n_dim": 10,
        "ub": 10000,
    },
    "Food": {
        "stim": "../Data/Behavioral Data/design_data_food.csv",
        "data": "../Data/Behavioral Data/data_analysis_food.csv",
        "n_dim": 14,
        "ub": 100,
    },
}



OPTIM_CONFIG = {
    "N_RESTARTS": 10, # number of random restarts for MLE
    "MAXITER": 5000,  # maximum iterations for optimization
    "FTOL": 1e-10,    # function value tolerance for convergence
    "GTOL": 1e-8,     # gradient norm tolerance for convergence
    "METHOD": 'L-BFGS-B', # optimization method
}     