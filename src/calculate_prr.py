import collections

import numpy as np


def compute_ABCD_one_drug(drug_exposures, drug_propensity_scores, all_outcomes,
                          bins=np.arange(0, 1.2, 0.2), seed=0):
    """
    Compute the propensity-score-matched numbers of reports with combinations
    of drug exposure and outcome occurrence. Note that for simplicity, "drug"
    refers to a specific single drug or a specific combination of drugs. When
    computing PRR for drug pairs (eg. TWOSIDES), then a "drug" refers to a
    particular combination of drugs, though all parameter types stay the same.

    Parameters
    ----------
    drug_exposures : scipy.sparse.csc_matrix
        Binary vector of exposures to the given drug
    drug_propensity_scores : numpy.ndarray
        Vector of propensity scores for the given drug (probability of exposure
        to the drug for a given record)
    all_outcomes : scipy.sparse.csc_matrix
        Matrix of reports (rows) by outcomes (columns)
    bins : numpy.ndarray
        Default is [0, 0.2, 0.4, 0.6, 0.8, 1], as per Rami's function,
        `run_one_prr`
    seed : int
        Random seed for sampling unexposed controls for each PSM bin

    Returns
    -------
    Tuple[numpy.ndarray, int, numpy.ndarray, int]
        A, A + B, C, C + D. A is a vector of outcomes, where each value is the
        number of drug exposed reports having the outcome. A + B is the total
        number of exposed reports. C is a vector of outcomes, where each value
        is the number of non-drug-exposed reports having the outcome using the
        binned propensity score matching procedure. C + D is the total number
        of the non-drug-exposed reports from the propensity score matching
        procedure.
    """
    # Find the (row) indices of reports exposed to the drug
    exposed_indices, _ = drug_exposures.nonzero()

    # A + B is the number exposed to the given drug
    n_exposed = len(exposed_indices)

    # A is the number exposed with the outcome.
    #  Here computing A for all outcomes simultaneously, so exposed_with_outcome
    #  is a vector where each index is an outcome and the value is the number
    # drug exposed with the outcome.
    exposed_with_outcome = all_outcomes[exposed_indices].sum(axis=0)
    exposed_with_outcome = np.array(exposed_with_outcome).flatten()

    # Default bins and this binning procedure were found in Rami's work.
    #  Unlike the paper, this method does not divide the region of overlap
    #  into 20 bins, though this method may be more appropriate for drug
    #  combinations, where we don't expect many people to have been exposed.
    binned_scores = np.digitize(drug_propensity_scores, bins=bins)
    exposed_bin_freq = collections.Counter(binned_scores[exposed_indices])

    # Set random seed for reproducible sampling
    np.random.seed(seed)

    # Sample (with replacement) 10x unexposed for each exposed (bin-wise)
    matched_unexposed_indices = list()
    for bin_number, num_exposed_bin in exposed_bin_freq.items():
        available_unexposed_indices = set(np.where(binned_scores == bin_number)[0]) \
            - set(exposed_indices)
        # Same as Rami's code - if bin contains only exposed reports, add the
        #  exposed reports to A and B, but add nothing to C and D from this bin
        if not available_unexposed_indices:
            continue

        unexposed_sample = np.random.choice(
            list(available_unexposed_indices),
            size=(10 * num_exposed_bin)
        )
        matched_unexposed_indices.extend(unexposed_sample)

    # C + D is the number of propensity matched reports unexposed to the drug
    # Should always be 10 * n_exposed, but re-compute to be safe
    n_unexposed = len(matched_unexposed_indices)

    # C is the number unexposed with the outcome
    unexposed_with_outcome = all_outcomes[matched_unexposed_indices].sum(axis=0)
    unexposed_with_outcome = np.array(unexposed_with_outcome).flatten()

    # Return A, A+B, C, C+D
    return exposed_with_outcome, n_exposed, unexposed_with_outcome, n_unexposed


def compute_prr(exposed_with_outcome, n_exposed, unexposed_with_outcome, n_unexposed):
    if (n_exposed == 0) or (n_unexposed == 0):
        PRR = np.empty(exposed_with_outcome.shape)
        PRR_error = np.full(exposed_with_outcome.shape, np.inf)
        return PRR, PRR_error

    with np.errstate(divide='ignore', invalid='ignore'):
        PRR = ((exposed_with_outcome / n_exposed) / (unexposed_with_outcome / n_unexposed))
        PRR_error = np.sqrt((1 / exposed_with_outcome) + (1 / unexposed_with_outcome)
                            - (1 / n_exposed) - (1 / n_unexposed))
    return PRR, PRR_error
