NUM_PARAMETERS = 11

def model(parameters, cues, ex_cues=None, ex_crit=None):
    """
    Compute predicted criterion estimate with distance-based extrapolation adjustment.
    """
    n_dim = cues.shape[1]
    p = 2.0

    # Reference distance
    ex_gap = np.abs(ex_cues[:, None, :] - ex_cues[None, :, :]) ** p
    ex_gap = (ex_gap.sum(axis=-1)) ** (1.0 / p)
    d_ref  = np.median(ex_gap[ex_gap > 0]) + 1e-9

    # Sensitivity
    c = np.exp(np.clip(parameters[0], -20.0, 20.0)) / d_ref

    # Attention weights
    logits = np.concatenate([np.zeros(1), parameters[1:n_dim]])
    logits_max = np.max(logits)
    logits_stable = logits - logits_max
    w_exp = np.exp(logits_stable)
    w = (w_exp / np.sum(w_exp)) * n_dim

    diff      = np.abs(cues[:, None, :] - ex_cues[None, :, :]) ** p
    distances = ((diff * w[None, None, :]).sum(axis=-1)) ** (1.0 / p)
    sim       = np.exp(-c * distances)

    sim_sum   = np.maximum(sim.sum(axis=1), 1e-12)
    pred_crit = (sim * ex_crit[None, :]).sum(axis=1) / sim_sum

    # Distance-based extrapolation adjustment
    min_distance = np.min(distances, axis=1)
    beta = parameters[10]
    pred_crit += beta * min_distance

    return pred_crit