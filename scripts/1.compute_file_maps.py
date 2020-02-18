import concurrent.futures
import functools
import pathlib
import sys
import tarfile
from typing import List, Tuple, Union

import pandas as pd
import tqdm

sys.path.insert(0, '../src/')
import utils  # noqa:E402


def n_drug_to_column_names(n: int) -> Tuple[str]:
    """
    For OFFSIDES, files are [drug, bootstrap_iteration, file_type,
    subfile_name, archive_path].

    For NSIDES, files are [drug_1, ..., drug_N, file_type, subfile_name,
    archive_path]
    """
    if n == 1:
        return ('drug', 'bootstrap')
    else:
        return tuple(f'drug_index_{i}' for i in range(1, n + 1))


def get_subfiles(archive_file_path: pathlib.Path,
                 n_drugs: int) -> List[List[Union[int, str]]]:
    """
    Find all score, interaction, and log files within a `.tar.gz` archive.
    Each subfile is represented as a list.
    """
    file_locations = list()
    try:
        tar = tarfile.open(archive_file_path, mode='r:gz')
        subfiles = tar.getnames()
    except tarfile.ReadError:
        return None
    except EOFError:
        return None

    for subfile in subfiles:
        file_name_match = utils.extract_filepath_info(subfile)
        if not file_name_match:
            raise ValueError(f'{archive_file_path.name} contained '
                             f'{subfile} not matched')
        file_locations.append([*file_name_match, subfile, archive_file_path])
    return file_locations


def compute_file_map(archives_path: pathlib.Path,
                     n_drugs: int) -> pd.DataFrame:
    """Compute a file map for an entire directory of archives"""
    get_subfiles_partial = functools.partial(get_subfiles, n_drugs=n_drugs)

    archive_files = list(archives_path.glob('scores_*.tgz'))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        file_locations = list(tqdm.tqdm(
            executor.map(get_subfiles_partial, archive_files),
            total=len(archive_files)
        ))

    # Flatten list of lists of tuples to a list of tuples
    flattened_file_locations = [l for l in file_locations if l is not None]
    flattened_file_locations = [i for l in flattened_file_locations for i in l]

    column_names = ['file_type', *n_drug_to_column_names(n_drugs), 'file_name',
                    'archive_file']

    # Instantiate DataFrame and save to disk immediately
    return pd.DataFrame(flattened_file_locations, columns=column_names)


def compute_all_filemaps():
    # # Path to where the `.tgz` archives are stored
    # archives_path = pathlib.Path('/data/archives/')
    # # Path where the file maps will be saved
    # meta_path = pathlib.Path('/data/meta')
    # # Compute and save OFFSIDES file map
    # offsides_file_map = compute_file_map(1, archives_path.joinpath('1/'))
    # offsides_file_map.to_csv(meta_path.joinpath('file_map_offsides.csv'),
    #                          index=False)
    # # Compute and save TWOSIDES file map
    # twosides_file_map = compute_file_map(2, archives_path.joinpath('2/'))
    # twosides_file_map.to_csv(meta_path.joinpath('file_map_twosides.csv'),
    #                          index=False)

    # Compute and save THREESIDES file map
    nsides_root = pathlib.Path('/data2/nsides/')
    threesides_file_map = compute_file_map(nsides_root.joinpath('dddi/'), 3)
    threesides_file_map.to_csv(
        nsides_root.joinpath('meta/file_map_twosides.csv'), index=False)


if __name__ == "__main__":
    compute_all_filemaps()
