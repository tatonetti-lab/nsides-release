import collections

import numpy as np


def compute_ABCD_one_drug(drug_exposures, drug_propensity_scores, all_outcomes,
                          bins=np.arange(0, 1.2, 0.2), seed=0):
    """
    Compute the propensity-score-matched numbers of reports with combinations
    of drug exposure and outcome occurrence.

    This procedure splits the range of potential propensity scores into `bins`.
    Controls (unexposed reports) are sampled to give 10 times the number of
    controls as cases (exposed reports), in each bin. If a bin contains only
    controls or only cases, then no reports are added from the bin.

    Note that for simplicity, "drug" refers to a specific single drug or a
    specific combination of drugs. When computing PRR for drug pairs
    (eg. TWOSIDES), then a "drug" refers to a particular combination of drugs,
    though all parameter types stay the same.

    Parameters
    ----------
    drug_exposures : scipy.sparse.csc_matrix
        Binary vector of exposures to the given drug. Shape is (n_reports x 1)
    drug_propensity_scores : numpy.ndarray
        Vector of propensity scores for exposure to the given drug. Shape is
        (n_reports x 1)
    all_outcomes : scipy.sparse.csc_matrix
        Matrix of reports (rows) by outcomes (columns). Shape is
        (n_reports x n_outcomes)
    bins : numpy.ndarray
        Default is [0, 0.2, 0.4, 0.6, 0.8, 1]
    seed : int
        Random seed for sampling unexposed controls for each PSM bin

    Returns
    -------
    Tuple[numpy.ndarray, int, numpy.ndarray, int]
        A, A + B, C, C + D.

        A is a vector of outcomes, where each value is the number of reports
        with that exposure having the outcome.
        A + B is the total number of exposed reports.
        C is a vector of outcomes, where each value is the number of
        non-drug-exposed reports having the outcome.
        C + D is the total number of the non-drug-exposed reports.
    """
    # Find the (row) indices of reports exposed to the drug
    exposed_indices, _ = drug_exposures.nonzero()

    # Default bins and this binning procedure were found in Rami's work.
    #  Unlike the paper, this method does not divide the region of overlap
    #  into 20 bins, though this method may be more appropriate for drug
    #  combinations, where we don't expect many people to have been exposed.
    binned_scores = np.digitize(drug_propensity_scores, bins=bins)
    exposed_bin_freq = collections.Counter(binned_scores[exposed_indices])

    # Set random seed for reproducible sampling
    np.random.seed(seed)

    # Sample (with replacement) 10x unexposed for each exposed (bin-wise)
    matched_exposed_indices = list()
    matched_unexposed_indices = list()
    for bin_number, num_exposed_bin in exposed_bin_freq.items():
        if num_exposed_bin == 0:
            continue

        # Indices of all reports in this bin
        reports_in_bin = np.where(binned_scores == bin_number)[0]
        reports_in_bin = set(reports_in_bin.tolist())

        # Indices of unexposed reports in this bin
        available_unexposed_indices = reports_in_bin - set(exposed_indices)
        if len(available_unexposed_indices) == 0:
            continue

        # Indices of exposed reports in this bin
        bin_exposed_indices = reports_in_bin.intersection(set(exposed_indices))
        matched_exposed_indices.extend(list(bin_exposed_indices))

        # The number of unexposed to sample
        num_unexposed = 10 * num_exposed_bin
        # Sample with replacement from unexposed indices
        unexposed_sample = np.random.choice(list(available_unexposed_indices),
                                            size=num_unexposed, replace=True)
        matched_unexposed_indices.extend(unexposed_sample)

    # A + B is the number exposed to the given drug
    n_exposed = len(matched_exposed_indices)

    # A is the number exposed with the outcome.
    #  Here computing A for all outcomes simultaneously, so exposed_with_outcome
    #  is a vector where each index is an outcome and the value is the number
    # drug exposed with the outcome.
    exposed_with_outcome = all_outcomes[matched_exposed_indices].sum(axis=0)
    exposed_with_outcome = np.array(exposed_with_outcome).flatten()

    # C + D is the number of propensity matched reports unexposed to the drug
    # Should always be 10 * n_exposed, but re-compute to be safe
    n_unexposed = len(matched_unexposed_indices)

    # C is the number unexposed with the outcome
    unexposed_with_outcome = all_outcomes[matched_unexposed_indices].sum(axis=0)
    unexposed_with_outcome = np.array(unexposed_with_outcome).flatten()

    # Return A, A+B, C, C+D
    return exposed_with_outcome, n_exposed, unexposed_with_outcome, n_unexposed


def compute_prr(exposed_with_outcome, n_exposed, unexposed_with_outcome, n_unexposed):
    """
    Compute PRR and PRR_error for a single drug. Uses A, B, C, and D as
    computed using `compute_ABCD_one_drug`.

    Parameters
    ----------
    exposed_with_outcome : numpy.ndarray
        Vector of outcomes, where each value specifies the number of reports
        exposed to the drug who had the outcome. These values are called A.
    n_exposed : int
        Number of reports exposed to the drug. This value is A + B, and it
        depends only on the drug of interest, not the outcome.
    unexposed_with_outcome : numpy.ndarray
        Vector of outcomes, where each value specifies the number of reports
        not exposed to the drug who had the outcome. These values are called C.
    n_unexposed : int
        Number of reports not exposed to the drug. This value is C + D, and it
        depends only on the drug of interest, not the outcome.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        PRR and PRR_error.
    """
    if (n_exposed == 0) or (n_unexposed == 0):
        PRR = np.empty(exposed_with_outcome.shape)
        PRR_error = np.full(exposed_with_outcome.shape, np.inf)
        return PRR, PRR_error

    with np.errstate(divide='ignore', invalid='ignore'):
        PRR = ((exposed_with_outcome / n_exposed) / (unexposed_with_outcome / n_unexposed))
        PRR_error = np.sqrt((1 / exposed_with_outcome) + (1 / unexposed_with_outcome)
                            - (1 / n_exposed) - (1 / n_unexposed))
    return PRR, PRR_error
