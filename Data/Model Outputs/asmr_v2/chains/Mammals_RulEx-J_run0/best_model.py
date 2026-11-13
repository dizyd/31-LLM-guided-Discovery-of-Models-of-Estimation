import numpy as np

NUM_PARAMETERS = 24

def model(parameters, cues, ex_cues=None, ex_crit=None):
    """
    Compute predicted criterion estimate.
    """
    n_dim = cues.shape[1]
    p = np.log1p(np.exp(np.clip(parameters[22], -30.0, 30.0)))  # Learnable distance metric parameter

    # Dynamic alpha based on distance to nearest exemplar
    distances = np.abs(cues[:, None, :] - ex_cues[None, :, :]) ** p
    distances = np.sum(distances, axis=-1)
    distances = distances ** (1.0 / p)
    nearest_distance = np.min(distances, axis=1)
    alpha = 1.0 / (1.0 + np.exp(-parameters[0] * nearest_distance))

    # Rule module (CAM)
    loc = np.mean(ex_crit)
    scale = np.std(ex_crit) + 1e-9
    cue_sd = np.std(ex_cues, axis=0) + 1e-9
    intercept = loc + parameters[1] * scale
    weights = parameters[2:2 + n_dim] * scale / cue_sd
    judge_r = intercept + cues @ weights

    # Exemplar module (nearest exemplar value)
    nearest_indices = np.argmin(distances, axis=1)
    judge_e = ex_crit[nearest_indices]

    # Boost nearest exemplar similarity (reused from original parameter[23])
    sim_boost = np.exp(np.clip(parameters[23], -20.0, 20.0))
    judge_e *= sim_boost  # Amplify exemplar influence

    # Combine rule and exemplar predictions
    pred_crit = alpha * judge_r + (1.0 - alpha) * judge_e
    pred_crit = np.log1p(np.exp(np.clip(pred_crit, -30.0, 30.0)))  # Ensure positive predictions

    return pred_crit