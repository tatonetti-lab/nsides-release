import concurrent.futures
import functools
import pathlib
import sys

import numpy as np
import scipy.sparse
import tqdm

sys.path.insert(0, '../src/')
import parallel_utils  # noqa:E402
import utils  # noqa:E402


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


def compute_prr_twosides(propensity_scores_path, prr_save_path,
                         report_exposure_matrix, report_outcome_matrix,
                         drug_id_vector, outcome_id_vector):
    computable_combinations = list(propensity_scores_path.glob('*.npy'))
    computable_combinations = [utils.extract_indices_twosides(path)
                               for path in computable_combinations]

    run_one_combination = functools.partial(
        parallel_utils.prr_one_combination,
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
            tqdm.tqdm(executor.map(run_one_combination, computable_combinations),
                    total=len(computable_combinations))
        )


def main():
    # User-specified directory paths
    meta_files_path = pathlib.Path('/data/meta/')
    propensity_scores_path = pathlib.Path('/data/scores/')

    prr_save_path = pathlib.Path('/data/prr/')
    prr_save_path.mkdir(exist_ok=True)

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

    compute_prr_offsides(propensity_scores_path.joinpath('1/'),
                         prr_save_path.joinpath('1/'),
                         report_exposure_matrix, report_outcome_matrix,
                         drug_id_vector, outcome_id_vector)

    # compute_prr_twosides(propensity_scores_path.joinpath('2/'),
    #                      prr_save_path.joinpath('2/'),
    #                      report_exposure_matrix, report_outcome_matrix,
    #                      drug_id_vector, outcome_id_vector)


if __name__ == "__main__":
    main()
