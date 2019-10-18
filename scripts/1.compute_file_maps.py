import concurrent.futures
import functools
import pathlib
import sys

import pandas as pd
import tqdm

sys.path.insert(0, '../src/')
import parallel_utils  # noqa:E402


def compute_offsides_map(archives_path, meta_path):
    get_offsides_subfiles = functools.partial(parallel_utils.get_subfiles,
                                              n_drugs=1)

    archive_files = list(archives_path.glob('1/scores_*.tgz'))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        file_locations = list(tqdm.tqdm(
            executor.map(get_offsides_subfiles, archive_files),
            total=len(archive_files)
        ))

    # Flatten the list of lists of tuples to a list of tuples
    file_locations = [i for l in file_locations for i in l]
    files_map = pd.DataFrame(file_locations,
                             columns=['drug', 'bootstrap', 'file_type',
                                      'file_name', 'archive_file'])
    files_map.to_csv(meta_path.joinpath('file_map_offsides.csv'), index=False)


def compute_twosides_map(archives_path, meta_path):
    get_twosides_subfiles = functools.partial(parallel_utils.get_subfiles,
                                              n_drugs=2)

    archive_files = list(archives_path.glob('2/scores_*.tgz'))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        file_locations = list(tqdm.tqdm(
            executor.map(get_twosides_subfiles, archive_files),
            total=len(archive_files)
        ))

    # Flatten the list of lists of tuples to a list of tuples
    flattened_file_locations = [l for l in file_locations if l is not None]
    flattened_file_locations = [i for l in flattened_file_locations for i in l]
    files_map = pd.DataFrame(flattened_file_locations,
                             columns=['drug_index_1', 'drug_index_2',
                                      'file_type', 'file_name', 'archive_file'])
    files_map.to_csv(meta_path.joinpath('file_map_twosides.csv'), index=False)


def main():
    # Path to where the `.tgz` archives are stored
    archives_path = pathlib.Path('/data/archives/')

    # Path where the file maps will be saved
    meta_path = pathlib.Path('/data/meta')

    # compute_offsides_map(archives_path, meta_path)
    compute_twosides_map(archives_path, meta_path)


if __name__ == "__main__":
    main()
