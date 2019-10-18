import concurrent.futures
import functools
import pathlib
import sys

import pandas as pd
import tqdm

sys.path.insert(0, '../src/')
import parallel_utils  # noqa:E402


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
        parallel_utils.compute_propensity_scores, files_map_df=files_map_df,
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
