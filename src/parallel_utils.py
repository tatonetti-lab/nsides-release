import tarfile

import numpy as np
import pandas as pd

import calculate_prr
import compute_scores


def compute_propensity_scores(drug_index, files_map_df, computed_scores_path,
                              temporary_directory):
    """
    Parameters
    ----------
    drug_index : int
    files_map_df : pandas.DataFrame
        DataFrame showing in which tar archives bootstrapped score and log
        files for each drug are located.
    computed_scores_path : pathlib.Path
        Path to directory where computed propensity scores (now averaged
        across bootstrap iterations) should be stored
    temporary_directory : pathlib.Path
        Directory where files should be extracted. All files will be fairly
        quickly removed after extraction, but this directory should be capable
        of storing at least a gigabyte. For that reason, the /tmp directory is
        not always appropriate and this must be user specified.

    Returns
    -------
    List[Tuple[int, int, float]]
        List of the AUC values for each bootstrap iteration. It is useful to
        keep and save these so that if the analysis must be repeated the AUC
        files need not be read in at all.
    """
    # Query a dataframe of files for the drug of interest
    drug_df = files_map_df.query(f'drug == {drug_index}')

    # Only open each tar file once, map paths to these open files
    tar_path_to_tar = {tar_file_path: tarfile.open(tar_file_path, mode='r:gz')
                       for tar_file_path in set(drug_df['archive_file_path'])}

    # Get tar members for relevant log files
    drug_df = (
        drug_df
        .assign(
            tar=lambda df: df['archive_file_path'].map(tar_path_to_tar),
            member=lambda df: df.apply(lambda row: row['tar'].getmember(row['file_name']), axis=1),
        )
    )

    # Load AUC values for each bootstrap iteration as a dict(iteration: auc)
    bootstrap_to_auc = compute_scores.get_drug_bootstrap_auc(drug_df, temporary_directory)
    drug_df['auc'] = drug_df['bootstrap'].map(bootstrap_to_auc)

    # Compute propensity scores. Uses computed AUCs to judge what iterations
    #  to include in the average.
    drug_scores = compute_scores.get_drug_scores(drug_df, temporary_directory)
    if drug_scores is None:
        return [(drug_index, None, None)]

    # Save computed (average) propensity scores
    np.savez_compressed(computed_scores_path.joinpath(f'{drug_index}.npz'),
                        scores=drug_scores)

    return [(drug_index, bootstrap, auc) for bootstrap, auc in bootstrap_to_auc.items()]


def load_scores(drug_index, n_rows, scores_path):
    """
    Parameters
    ----------
    drug_index : int
    n_rows : int
    scores_path : pathlib.Path
        Path to the directory where propensity scores for each drug are stored
        as <drug index>.npz files.

    Returns
    -------
    numpy.ndarray
    """
    score_path = scores_path.joinpath(f'{drug_index}.npz')
    scores_item = np.load(score_path)

    # Slice to the relevant number of reports (originally 4_838_588, not 4_694_086)
    scores = scores_item['scores']
    scores = scores[:n_rows]
    return scores


def prr_one_drug(drug_index, all_exposures, all_outcomes, n_reports,
                 drug_id_vector, outcome_id_vector, scores_path):
    """
    Helper function to compute and save disproportionality statistics for a
    given drug. To use with `concurrent.futures` most easily, the user should
    load necessary arrays and create a `functools.partial` function so that
    the resulting function requires only `drug_index`. This partial function
    can then be mapped to an iterable of integers for each drug index and
    easily parallelized using `concurrent.futures.ProcessPoolExecutor`.
    """
    scores = load_scores(drug_index, n_reports, scores_path)
    drug_exposures = all_exposures[:, drug_index]
    A, a_plus_b, C, c_plus_d = calculate_prr.compute_ABCD_one_drug(drug_exposures,
                                                                   scores,
                                                                   all_outcomes)
    prr, prr_error = calculate_prr.compute_prr(A, a_plus_b, C, c_plus_d)

    drug_df = (
        pd.DataFrame()
        .assign(
            outcome_id=outcome_id_vector,
            A=A,
            B=a_plus_b - A,
            C=C,
            D=c_plus_d - C,
            PRR=prr,
            PRR_error=prr_error,
            drug_id=drug_id_vector[drug_index],
        )
        # The first entry in the outcome vector is `None`
        .query('not outcome_id.isnull()')
        .filter(items=['drug_id', 'outcome_id', 'A', 'B', 'C', 'D',
                       'PRR', 'PRR_error', 'mean'])
    )
    drug_df.to_csv(f'../data/prr/{drug_index}.csv.xz', index=False, compression='xz')
