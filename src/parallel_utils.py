import pandas as pd

import calculate_prr
import utils


def prr_one_drug(drug_index, all_exposures, all_outcomes, n_reports,
                 drug_id_vector, outcome_id_vector, scores_path, save_path):
    """
    Helper function to compute and save disproportionality statistics for a
    given drug. To use with `concurrent.futures` most easily, the user should
    load necessary arrays and create a `functools.partial` function so that
    the resulting function requires only `drug_index`. This partial function
    can then be mapped to an iterable of integers for each drug index and
    easily parallelized using `concurrent.futures.ProcessPoolExecutor`.
    """
    scores = utils.load_scores_offsides(drug_index, n_reports, scores_path)
    drug_exposures = all_exposures[:, drug_index]

    drug_df = _prr_helper(scores, drug_exposures, all_outcomes,
                          outcome_id_vector)

    drug_df = (
        drug_df
        .assign(drug_id=drug_id_vector[drug_index])
        .filter(items=['drug_id', 'outcome_id', 'A', 'B', 'C', 'D',
                       'PRR', 'PRR_error'])
    )
    drug_df.to_csv(save_path.joinpath(f'{drug_index}.csv.xz'), index=False,
                   compression='xz')


def prr_one_combination(drug_indices, all_exposures, all_outcomes, n_reports,
                        drug_id_vector, outcome_id_vector, scores_path, save_path):
    """
    Parameters
    ----------
    drug_indices : List[int] (or subscriptable array of int)
        Assumed to be sorted from smallest to largest, as this is how the files
        are named. Though note, the columns in the returned pandas.DataFrame
        will not necessarily be sorted so that drug_1 < drug_2, etc. This is
        because the values in these drug columns are IDs, and we don't want
        to enforce a sorting method on IDs which may not be integers.
    Other parameters are identical to the function for a single drug.
    """
    # For some reason a number of files fail to load or don't contain data, etc.
    try:
        scores, indices_string = utils.load_scores_nsides(drug_indices, n_reports,
                                                          scores_path)
    except:  # noqa:E722
        return

    drug_exposures = utils.compute_multi_exposure(drug_indices, all_exposures)

    drug_df = _prr_helper(scores, drug_exposures, all_outcomes,
                          outcome_id_vector)

    # Add IDs of drugs as columns drug_1, drug_2, ..., drug_n
    drug_columns = list()
    for i, drug_index in enumerate(drug_indices):
        drug_df[f'drug_{i+1}'] = drug_id_vector[drug_index]
        drug_columns.append(f'drug_{i+1}')

    drug_df = drug_df.filter(items=[*drug_columns, 'outcome_id', 'A', 'B', 'C',
                                    'D', 'PRR', 'PRR_error'])
    drug_df.to_csv(save_path.joinpath(indices_string + '.csv.xz'),
                   index=False, compression='xz')


def _prr_helper(scores, drug_exposures, all_outcomes, outcome_id_vector):
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
        )
        # The first entry in the outcome vector is `None`
        .query('not outcome_id.isnull()')
    )
    return drug_df
