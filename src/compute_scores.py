import os
import tarfile

import numpy as np


def extract_drug_files(tar_path_to_members, extract_dir):
    """
    Extract member files from a number of tar archives, returning extracted
    paths.

    Parameters
    ----------
    tar_path_to_members : Dict[pathlib.Path, List[tarfile.TarInfo]]
        Dictionary mapping tar file paths to the member files to extract
    extract_dir : pathlib.Path
        The directory into which files should be extracted

    Returns
    -------
    Dict[tarfile.TarInfo, pathlib.Path]
        A map from tar member files to their file paths once extracted
    """
    file_name_to_extracted_path = dict()
    for archive_path, member_files in tar_path_to_members.items():
        tar = tarfile.open(archive_path)
        tar.extractall(path=extract_dir, members=member_files)
        file_name_to_extracted_path.update(
            {member: extract_dir.joinpath(member.name) for member in member_files}
        )
    return file_name_to_extracted_path


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


def get_drug_bootstrap_auc(drug_df, temporary_directory):
    """
    Load the AUC values for all bootstrap iterations of a given drug's scores

    Parameters
    ----------
    drug_df : pandas.DataFrame
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
    log_files_df = drug_df.query('file_type == "log"')

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
    log_file_to_extracted_path = extract_drug_files(archive_to_logs, temporary_directory)
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
    score_file_extracted_paths = extract_drug_files(archive_to_scores, temporary_directory)
    if len(score_file_extracted_paths.values()) == 0:
        return None

    # Compute the propensity scores for a drug as the average over bootstrap
    #  iterations with auc > 0.5
    drug_scores = compute_average_propensity_score(score_file_extracted_paths.values())

    # Delete extracted files from disk
    for score_path in score_file_extracted_paths.values():
        os.remove(score_path)

    return drug_scores
