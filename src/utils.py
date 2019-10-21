import re
import tarfile

import numpy as np


def extract_indices(filename):
    match = re.match(r'(?:.+_lrc_)([0-9]+)(?:__)([0-9]+)(?:\.npy)', filename)
    if match:
        bootstrap, drug = match.groups()
        return int(drug), int(bootstrap)
    else:
        return False, False


def extract_indices_twosides(filename, original=True):
    if original:
        match_string = r'(?:.+__)([0-9]+)(?:_)([0-9]+)(?=\.npy)'
    else:
        match_string = r'([0-9]+)(?:_)([0-9]+)(?=\.npy)'
    drug_a, drug_b = re.match(match_string, filename).groups()
    drug_a, drug_b = int(drug_a), int(drug_b)
    return drug_a, drug_b


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


def load_scores_offsides(drug_index, n_rows, scores_path):
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


def load_scores_nsides(drug_indices, n_rows, scores_path):
    indices_string = '_'.join(map(str, drug_indices))
    score_path = scores_path.joinpath(indices_string + '.npy')
    scores = np.load(score_path)

    # Slice to the relevant number of reports (originally 4_838_588, not 4_694_086)
    scores = scores[:n_rows]
    return scores, indices_string


def compute_multi_exposure(drug_indices, all_exposures):
    """
    Computes a binary vector for multiple exposures. A report has a 1 if it
    was exposed to all drugs in `drug_indices` and 0 otherwise.
    """
    drug_exposures = None
    for drug_index in drug_indices:
        exposure = all_exposures[:, drug_index]
        if drug_exposures is None:
            drug_exposures = exposure
        else:
            drug_exposures = drug_exposures.multiply(exposure)
    return drug_exposures
