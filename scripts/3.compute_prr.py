import concurrent.futures
import functools
import os
import pathlib
import re
import sys
import tarfile

import numpy as np
import pandas as pd
import scipy.sparse
import tqdm

sys.path.insert(0, '../src/')
import parallel_utils  # noqa:E402
import utils  # noqa:E402


def extract_scores_twosides(tar_file_path, computed_scores_path):
    """Extract all propensity score files from a tar file at the given path"""
    try:
        tar = tarfile.open(tar_file_path, mode='r:gz')
        members = tar.getmembers()
    except:  # noqa:E722
        return None

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


def compute_prr_offsides(propensity_scores_path, prr_save_path,
                         report_exposure_matrix, report_outcome_matrix,
                         drug_id_vector, outcome_id_vector):
    computable_drugs = list(propensity_scores_path.glob('*.npz'))
    computable_drugs = sorted([int(drug.stem) for drug in computable_drugs])

    run_one_drug = functools.partial(
        parallel_utils.prr_one_drug,
        all_exposures=report_exposure_matrix,
        all_outcomes=report_outcome_matrix,
        n_reports=report_exposure_matrix.shape[0],
        drug_id_vector=drug_id_vector,
        outcome_id_vector=outcome_id_vector,
        scores_path=propensity_scores_path,
        save_path=prr_save_path,
    )

    # Compute and save disproportionality files (one for each drug)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(  # noqa: F841
            tqdm.tqdm(executor.map(run_one_drug, computable_drugs),
                    total=len(computable_drugs))
        )


def prr_one_archive_twosides(archive_path, file_map, extract_dir,
                             report_exposure_matrix, report_outcome_matrix,
                             drug_id_vector, outcome_id_vector, prr_save_path):
    # Extract propensity scores from archive
    extracted_paths = parallel_utils.extract_scores_twosides(archive_path,
                                                             extract_dir)

    if extracted_paths is None:
        return

    # Get indices of drug combinations stored in the archive
    drug_indices = [
        utils.extract_indices_twosides(file.name, original=False)
        for file in extracted_paths
    ]

    prr_one_combo = functools.partial(
        parallel_utils.prr_one_combination,
        all_exposures=report_exposure_matrix,
        all_outcomes=report_outcome_matrix,
        n_reports=report_exposure_matrix.shape[0],
        drug_id_vector=drug_id_vector,
        outcome_id_vector=outcome_id_vector,
        scores_path=extract_dir,
        save_path=prr_save_path,
    )

    # Compute PRR et al. for each drug combination
    for indices in drug_indices:
        prr_one_combo(indices)

    # Delete extracted files
    list(map(os.remove, extracted_paths))


def compute_prr_twosides(archives_path, file_map, extract_dir,
                         report_exposure_matrix, report_outcome_matrix,
                         drug_id_vector, outcome_id_vector, prr_save_path):
    archive_paths = list(archives_path.glob('scores_*.tgz'))

    run_one_archive = functools.partial(
        prr_one_archive_twosides,
        file_map=file_map, extract_dir=extract_dir,
        report_exposure_matrix=report_exposure_matrix,
        report_outcome_matrix=report_outcome_matrix,
        drug_id_vector=drug_id_vector,
        outcome_id_vector=outcome_id_vector,
        prr_save_path=prr_save_path
    )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(  # noqa: F841
            tqdm.tqdm(executor.map(run_one_archive, archive_paths),
                      total=len(archive_paths))
        )


def main():
    # User-specified directory paths
    meta_files_path = pathlib.Path('/data/meta/')
    propensity_scores_path = pathlib.Path('/data/scores/')
    twosides_archives_path = pathlib.Path('/data/archives/2/')
    temp_extract_dir = pathlib.Path('/data/extract_dir/')
    temp_extract_dir.mkdir(exist_ok=True)

    prr_save_path = pathlib.Path('/data/prr/')
    prr_save_path.mkdir(exist_ok=True)
    prr_save_path.joinpath('1/').mkdir(exist_ok=True)
    prr_save_path.joinpath('2/').mkdir(exist_ok=True)

    # Load matrices of reports by exposures and outcomes
    report_exposure_matrix = scipy.sparse.load_npz(
        meta_files_path.joinpath('drug_exposure_matrix.npz')
    )
    report_outcome_matrix = scipy.sparse.load_npz(
        meta_files_path.joinpath('outcome_matrix.npz')
    )

    print(f'Exposures: {report_exposure_matrix.shape},'
          f' Outcomes: {report_outcome_matrix.shape}')

    # Load vectors of the ids at each index for exposures and outcomes
    drug_id_vector = np.load(
        meta_files_path.joinpath('drug_id_vector.npy')
    ).astype(str)
    outcome_id_vector = np.load(
        meta_files_path.joinpath('outcome_id_vector.npy')
    )

    # Load TWOSIDES file map
    twosides_file_map = pd.read_csv(meta_files_path
                                    .joinpath('file_map_twosides.csv'))

    compute_prr_offsides(propensity_scores_path.joinpath('1/'),
                         prr_save_path.joinpath('1/'),
                         report_exposure_matrix, report_outcome_matrix,
                         drug_id_vector, outcome_id_vector)

    compute_prr_twosides(twosides_archives_path, twosides_file_map,
                         temp_extract_dir, report_exposure_matrix,
                         report_outcome_matrix, drug_id_vector,
                         outcome_id_vector, prr_save_path.joinpath('2/'))


if __name__ == "__main__":
    main()
