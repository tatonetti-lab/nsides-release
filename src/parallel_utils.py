import re
import tarfile

import numpy as np
import pandas as pd

import calculate_prr
import compute_scores
import utils


def get_subfiles(archive_file_path, n_drugs):
    file_locations = list()
    try:
        tar = tarfile.open(archive_file_path, mode='r:gz')
        subfiles = tar.getnames()
    except tarfile.ReadError:
        return None
    except EOFError:
        return None

    if n_drugs == 1:
        for subfile in subfiles:
            if 'interaction' in subfile:
                drug = re.match(r'(?:interactions__)([0-9]+)(?:\.npy)', subfile)
                if not drug:
                    raise ValueError(f'{archive_file_path.name} contained {subfile} not matched')
                drug = drug.group(1)
                bootstrap = None
                file_type = 'interaction'
            else:
                drug, bootstrap = utils.extract_indices(subfile)
                file_type = re.match('^[a-z]+(?=_.+)', subfile).group()
            file_locations.append([drug, bootstrap, file_type, subfile,
                                   archive_file_path.name])
    elif n_drugs == 2:
        for subfile in subfiles:
            file_name_match = re.match(r'([a-z]+)(?:.*?__)([0-9]+)(?:_)([0-9]+)(?=\.npy)', subfile)
            if not file_name_match:
                raise ValueError(f'{archive_file_path.name} contained {subfile} not matched')
            file_type, drug_1, drug_2 = file_name_match.groups()
            drug_1, drug_2 = int(drug_1), int(drug_2)
            file_locations.append([drug_1, drug_2, file_type, subfile,
                                   archive_file_path.name])
    return file_locations


def compute_propensity_scores(drug_index, files_map_df, computed_scores_path,
                              temporary_directory):
    # TODO: Refactor this into compute_scores.py and only wrap it here.
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


def extract_scores_twosides(tar_file_path, computed_scores_path):
    """Extract all propensity score files from a tar file at the given path"""
    tar = tarfile.open(tar_file_path, mode='r:gz')
    members = tar.getmembers()
    scores_members = [member for member in members if 'score' in member.name]
    tar.extractall(path=computed_scores_path, members=scores_members)

    # Rename files from 'scores_lrc__0_1.npy' to '0_1.npy'
    extracted_paths = list()
    for member in scores_members:
        path = computed_scores_path.joinpath(member.name)
        assert path.is_file()
        new_name = re.match(r'(?:.+__)([0-9_]+\.npy)', member.name).group(1)
        new_path = path.parent.joinpath(new_name)
        path.rename(new_path)
        extracted_paths.append(new_path)
    return extracted_paths


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
    scores, indices_string = utils.load_scores_nsides(drug_indices, n_reports,
                                                      scores_path)
    drug_exposures = utils.compute_multi_exposure(drug_indices, all_exposures)

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
    )

    # Add IDs of drugs as columns drug_1, drug_2, ..., drug_n
    drug_columns = list()
    for i, drug_index in enumerate(drug_indices):
        drug_df[f'drug_{i+1}'] = drug_id_vector[drug_index]
        drug_columns.append(f'drug_{i+1}')

    drug_df = (
        drug_df
        # The first entry in the outcome vector is `None`
        .query('not outcome_id.isnull()')
        .filter(items=[*drug_columns, 'outcome_id', 'A', 'B', 'C', 'D',
                       'PRR', 'PRR_error', 'mean'])
    )
    drug_df.to_csv(save_path.joinpath(indices_string + '.csv.xz'),
                   index=False, compression='xz')
