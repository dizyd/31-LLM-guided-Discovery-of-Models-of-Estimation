model_string_CAM = '''

NUM_PARAMETERS = 12
BOUNDS         = [(-10000/3, 10000/3)] * (10 + 1) + [(1e-3, 10000)]

def model(parameters, cues, ex_cues=None, ex_crit=None):
    """
    Compute predicted criterion estimate.

    Parameters
    ----------
    parameters : np.ndarray of shape (num_parameters,)
        Model parameters. intercept first, then one weight per cue

    cues : np.ndarray of shape  (n_trials, n_dim)
        Feature matrix of stimuli accross trials

    Returns
    -------
    pred_crit : np.ndarray of shape (n_trials,)
        Predicted criterion estimates for each trial, clipped to the range [0, 10000]
    """
    
    n_dim      = cues.shape[1]
    w          = parameters[:n_dim + 1]      # intercept + weights
    sigma      = abs(parameters[-1])            

    pred_crit  = w[0] + cues @ w[1:]
    pred_crit  = np.clip(pred_crit, 0, 10000)

    return pred_crit, sigma'''


model_string_GCM = '''
NUM_PARAMETERS = 12
BOUNDS         = [(1e-4, 100)] + [(0, 10)] * 10 + [(1e-3, 10000)]

def model(parameters, cues, ex_cues, ex_crit, p = 2):
    """
    Compute predicted criterion estimate.

    Parameters
    ----------
    parameters : np.ndarray of shape (num_parameters,)
        Model parameters. sensitivity first, then one weight per cue

     cues : np.ndarray of shape  (n_trials, n_dim)
        Feature matrix of stimuli accross trials

    ex_cues    : np.ndarray of shape (n_exemplars, n_dim)
        Feature matrix of exemplars

    ex_crit    : np.ndarray of shape (n_exemplars,)
        Criterion values for each exemplar

    Returns
    -------
    pred_crit : np.ndarray of shape (n_trials,)
        Predicted criterion estimates for each trial, clipped to the range [0, 10000]
    """

    n_dim     = cues.shape[1]
    c         = abs(parameters[0])            # sensitivity
    w         = parameters[1:n_dim + 1]       # feature weights
    sigma     = abs(parameters[-1])   



    w = np.abs(w)
    w = w / (w.sum() + 1e-12) * n_dim         # normalise to simplex

    diff      = np.power((cues[:, np.newaxis, :] - ex_cues[np.newaxis, :, :]), p)
    distances = np.power((diff * w[np.newaxis, np.newaxis, :]).sum(axis=-1), 1/p)
    sim       = np.exp(-c * distances)                                       

    sim_sum   = sim.sum(axis=1, keepdims=True)
    sim_sum   = np.where(sim_sum == 0, 1e-12, sim_sum)

    pred_crit = (sim * ex_crit[np.newaxis, :]).sum(axis=1) / sim_sum.squeeze()
    pred_crit = np.clip(pred_crit, 0, 10000) 

    return pred_crit, sigma'''