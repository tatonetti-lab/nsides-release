import concurrent.futures
import functools
import os
import pathlib
import sys
import tarfile

import numpy as np
import scipy.sparse
import tqdm

sys.path.insert(0, '../src/')
import parallel_utils  # noqa:E402
import utils  # noqa:E402


def extract_scores_nsides(tar_file_path, computed_scores_path):
    """Extract all propensity score files from a tar file at the given path"""
    try:
        tar = tarfile.open(tar_file_path, mode='r:gz')
        members = tar.getmembers()
    except:  # noqa:E722
        return None

    scores_members = [member for member in members if 'score' in member.name]
    tar.extractall(path=computed_scores_path, members=scores_members)

    return [
        computed_scores_path.joinpath(member.name) for member in scores_members
    ]


def load_finished_archives(path):
    scores = set()
    with open(path, 'r') as f:
        for line in f.readlines():
            scores.add(line.strip())
    return scores


def prr_one_archive_nsides(archive_path, extract_dir, report_exposure_matrix,
                           report_outcome_matrix, drug_id_vector,
                           outcome_id_vector, prr_save_path, complete_log,
                           error_log):
    # Extract propensity scores from archive
    extracted_paths = extract_scores_nsides(archive_path, extract_dir)

    if extracted_paths is None:
        with open(error_log, 'a') as f:
            f.write(f'{archive_path}\n')
        return

    # Get indices of drug combinations stored in the archive
    drug_indices = [
        utils.extract_filepath_info(file.name) for file in extracted_paths
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
    with open(complete_log, 'a') as f:
        f.write(f'{archive_path}\n')


def main():
    meta_files_path = pathlib.Path('/data1/home/mnz2108/git/nsides-release/'
                                   'data/meta_formatted/')
    archives_path = pathlib.Path('/data2/nsides/dddi/')
    outputs_path = pathlib.Path('/data2/nsides/dddi_prr/')
    extract_path = pathlib.Path('/data2/nsides/temp/')

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

    # Get the archives that have already been computed
    success_log_path = '../log/threesides_success_log.log'
    finished_archives = load_finished_archives(success_log_path)
    error_log_path = '../log/threesides_error_log.log'

    # Load score archives that have yet to be computed
    score_archive_paths = [path for path in archives_path.glob('scores_*.tgz')
                           if path.as_posix() not in finished_archives]

    run_one_archive = functools.partial(
        prr_one_archive_nsides,
        extract_dir=extract_path,
        report_exposure_matrix=report_exposure_matrix,
        report_outcome_matrix=report_outcome_matrix,
        drug_id_vector=drug_id_vector,
        outcome_id_vector=outcome_id_vector,
        prr_save_path=outputs_path,
        complete_log=success_log_path,
        error_log=error_log_path,
    )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(  # noqa: F841
            tqdm.tqdm(executor.map(run_one_archive, score_archive_paths),
                      total=len(score_archive_paths))
        )


if __name__ == "__main__":
    main()
