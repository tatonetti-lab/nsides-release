import concurrent.futures
import functools
import os
import pathlib
import tarfile
import sys

import numpy as np
import pandas as pd
import tqdm

sys.path.insert(0, '../src/')
import utils  # noqa:E402


def get_drug_bootstrap_auc(drug_file_map_df, temporary_directory):
    """
    Load the AUC values for all bootstrap iterations of a given drug's scores.

    Parameters
    ----------
    drug_file_map_df : pandas.DataFrame
        DataFrame giving information about all files related to a given drug.
        Required columns are `file_type` (either `"log"` or `"scores"`),
        `archive_file_path` (the tar archive in which the file is located, a
        pathlib.Path), `member` (tarfile.TarInfo member for the file), and
        `bootstrap` (`int` giving bootstrap iteration).
    temporary_directory : pathlib.Path
        Where to extract log files. Files are later removed.

    Returns
    -------
    Dict[int, float]
        Map of bootstrap iteration to corresponding propensity score AUC
    """
    # Get the relevant log file rows for this drug
    # DataFrame query is needed twice, so compute only once
    log_files_df = drug_file_map_df.query('file_type == "log"')

    # Get files as {archive_file: [log_file, log_file, ...]} with archive_file
    #  as a pathlib.Path and log_files as tarfile.TarInfo objects
    archive_to_logs = (
        log_files_df
        .groupby('archive_file_path')['member']
        .apply(list)
        .to_dict()
    )

    # Extract all log files associated with the drug, across all archives that
    #  contains log files
    log_file_to_extracted_path = utils.extract_drug_files(archive_to_logs,
                                                          temporary_directory)
    log_files_df = log_files_df.assign(
        extracted_path=lambda df: df['member'].map(log_file_to_extracted_path),
    )

    # Read the log files into a dict like {bootstrap_index: auc_value}
    #  where bootstrap_index is an integer and auc_value is a float.
    #  Remove files once read.
    bootstrap_to_auc = dict()
    for bootstrap, file_path in log_files_df[['bootstrap', 'extracted_path']].values.tolist():
        log_info = np.load(file_path, allow_pickle=True, encoding='latin1')
        bootstrap_to_auc[bootstrap] = log_info.item()['auc']
        os.remove(file_path)

    return bootstrap_to_auc


def compute_average_propensity_score(score_file_paths):
    """Load score files and return their elementwise average"""
    drug_scores = None
    for score_file_path in score_file_paths:
        bootstrap_scores = np.load(score_file_path)
        if drug_scores is None:
            drug_scores = bootstrap_scores
        else:
            drug_scores += bootstrap_scores
    return drug_scores / len(score_file_paths)


def get_drug_scores(drug_df, temporary_directory):
    """
    Compute the average propensity scores given a compressed tar archive.
    Unlike `get_drug_bootstrap_auc`, here `drug_df` must also have `auc` values.

    Parameters
    ----------
    drug_df : pandas.DataFrame
        DataFrame giving information about all files related to a given drug.
        Required columns are `file_type` (either `"log"` or `"scores"`),
        `archive_file_path` (the tar archive in which the file is located, a
        pathlib.Path), `member` (tarfile.TarInfo member for the file), and
        `auc` (`float`).
    temporary_directory : pathlib.Path
        Where to extract score files. Files are later removed.

    Returns
    -------
    numpy.ndarray
        Propensity scores for the given drug
    """
    # Find score files corresponding to bootstrap iterations with auc > 0.5
    # Get tarfile members for relevant score files and find where they are stored.
    # Get files as {archive_file: [score_file, score_file, ...]} with archive_file
    #  as a pathlib.Path and score_files as tarfile.TarInfo objects
    archive_to_scores = (
        drug_df
        .query('file_type == "scores" & auc > 0.5')
        .groupby('archive_file_path')['member']
        .apply(list)
        .to_dict()
    )

    # Extract all score files associated with the drug, across all archives that
    #  contains score files
    score_file_extracted_paths = utils.extract_drug_files(archive_to_scores,
                                                          temporary_directory)
    if len(score_file_extracted_paths.values()) == 0:
        return None

    # Compute the propensity scores for a drug as the average over bootstrap
    #  iterations with auc > 0.5
    drug_scores = compute_average_propensity_score(score_file_extracted_paths.values())

    # Delete extracted files from disk
    for score_path in score_file_extracted_paths.values():
        os.remove(score_path)

    return drug_scores


def compute_propensity_scores_one_drug(drug_index, files_map_df,
                                       computed_scores_path,
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
        List of (drug_index, bootstrap, auc) tuples. It is useful to keep and
        save these so that if the analysis must be repeated then the AUC files
        need not be read in at all.
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
    bootstrap_to_auc = get_drug_bootstrap_auc(drug_df, temporary_directory)
    drug_df['auc'] = drug_df['bootstrap'].map(bootstrap_to_auc)

    # Compute propensity scores. Uses computed AUCs to judge what iterations
    #  to include in the average.
    drug_scores = get_drug_scores(drug_df, temporary_directory)
    if drug_scores is None:
        return [(drug_index, None, None)]

    # Save computed (average) propensity scores
    np.savez_compressed(computed_scores_path.joinpath(f'{drug_index}.npz'),
                        scores=drug_scores)

    return [(drug_index, bootstrap, auc) for bootstrap, auc in bootstrap_to_auc.items()]


def compute_propensity_scores_offsides(meta_files_path, archives_path,
                                       computed_scores_path, temp_extract_dir):
    """Compute PS by averaging bootstrap iterations"""
    files_map_df = (
        pd.read_csv(meta_files_path.joinpath('file_map_offsides.csv'))
        .assign(
            archive_file_path=lambda df: df['archive_file'].apply(archives_path.joinpath),
        )
    )
    drugs = sorted(set(files_map_df['drug'].astype(int)))
    compute_scores_partial = functools.partial(
        compute_propensity_scores_one_drug, files_map_df=files_map_df,
        temporary_directory=temp_extract_dir,
        computed_scores_path=computed_scores_path
    )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        all_aucs = list(tqdm.tqdm(
            executor.map(compute_scores_partial, drugs),
            total=len(drugs)
        ))

    # Flatten the list of lists of tuples to a list of tuples
    all_aucs = [i for l in all_aucs for i in l]
    all_auc_df = pd.DataFrame(all_aucs, columns=['drug', 'bootstrap', 'auc'])
    all_auc_df.to_csv(meta_files_path.joinpath('offsides_bootstrap_auc.csv'),
                      index=False)


def main():
    # User-specified directory paths
    meta_files_path = pathlib.Path('/data/meta/')
    archives_path = pathlib.Path('/data/archives/1/')

    # Directory for extracting and temporarily storing files
    temp_extract_dir = pathlib.Path('/data/extract_dir/')
    temp_extract_dir.mkdir(exist_ok=True)

    # Directory where computed score files will be stored
    computed_scores_path = pathlib.Path('/data/scores/1/')
    computed_scores_path.mkdir(parents=True, exist_ok=True)

    compute_propensity_scores_offsides(meta_files_path,
                                       archives_path,
                                       computed_scores_path,
                                       temp_extract_dir)


if __name__ == "__main__":
    main()
