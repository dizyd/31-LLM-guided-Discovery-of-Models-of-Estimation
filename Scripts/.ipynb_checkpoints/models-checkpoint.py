import numpy as np

# NUM_PARAMETERS = 1 + N_CUES   # intercept + cue weights

def model_CAM(parameters, cues, ub, ex_cues=None, ex_crit=None):
    """
    Compute predicted criterion estimate according to the Cue Abstraction Model (linear rule-based).

    Parameters
    ----------
    parameters : np.ndarray of shape (num_parameters,)
        Model parameters. intercept first, then one weight per cue

    cues : np.ndarray of shape  (n_trials, n_dim)
        Feature matrix of stimuli accross trials

    ub : np.ndarray of shape (1,)
        Upper bounds for the predicted criterion estimates

    Returns
    -------
    pred_crit : np.ndarray of shape (n_trials,)
        Predicted criterion estimates for each trial, clipped to the range [0, ub]
    """
    
    n_dim      = cues.shape[1]
    w          = parameters[:n_dim + 1]      # intercept + weights
    sigma      = abs(parameters[-1])            

    pred_crit  = w[0] + cues @ w[1:]
    pred_crit  = np.clip(pred_crit, 0, ub)

    return pred_crit, sigma



# NUM_PARAMETERS = 1 + N_CUES   # sensitivity parameter + cue weights

def model_GCM(parameters, cues, ub, ex_cues, ex_crit, p = 2):
    """
    Compute predicted criterion estimate according to the Generalized Context Model (exemplar-based).

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

    ub : np.ndarray of shape (1,)
        Upper bounds for the predicted criterion estimates

    Returns
    -------
    pred_crit : np.ndarray of shape (n_trials,)
        Predicted criterion estimates for each trial, clipped to the range [0, ub]
    """

    n_dim     = cues.shape[1]
    c         = abs(parameters[0])            # sensitivity
    w         = parameters[1:n_dim + 1]       # feature weights
    sigma      = abs(parameters[-1])   



    w = np.abs(w)
    w = w / (w.sum() + 1e-12) * n_dim         # normalise to simplex

    # We compute the distance between each trial and each exemplar, weighted by the feature weights,
    # and then transform this distance into a similarity using an exponential decay function.

    diff      = np.power((cues[:, np.newaxis, :] - ex_cues[np.newaxis, :, :]), p)
    distances = np.power((diff * w[np.newaxis, np.newaxis, :]).sum(axis=-1), 1/p)
    sim       = np.exp(-c * distances)                                       

    sim_sum   = sim.sum(axis=1, keepdims=True)
    sim_sum   = np.where(sim_sum == 0, 1e-12, sim_sum)

    pred_crit = (sim * ex_crit[np.newaxis, :]).sum(axis=1) / sim_sum.squeeze()
    pred_crit = np.clip(pred_crit, 0, ub)

    return pred_crit, sigma