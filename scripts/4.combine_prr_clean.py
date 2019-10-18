import os
import pathlib
import shutil
import tarfile

import pandas as pd
import tqdm


def combine_prr_files(prr_files_path, save_path):
    files = list(prr_files_path.glob('*.csv.xz'))

    header = True
    for file_path in tqdm.tqdm(files):
        (
            pd.read_csv(file_path)
            .to_csv(save_path, index=False, header=header, mode='a',
                    compression='xz')
        )
        header = False
        os.remove(file_path)


def combine_files_to_archive(file_paths, save_path):
    tar = tarfile.open(save_path, "w:xz")
    for file_path in tqdm.tqdm_notebook(file_paths):
        tar.add(file_path, arcname=file_path.name)
    tar.close()


def main():
    # shutil.rmtree('/data/extract_dir/')

    data_path = pathlib.Path('/data/')
    tables_path = pathlib.Path('/data/tables/')
    output_archive_path = pathlib.Path('/data/output_archives/')
    output_archive_path.mkdir(exist_ok=True)

    # Combine OFFSIDES PRR files and save to a single table file
    # combine_prr_files(data_path.joinpath('prr/1/'),
    #                   tables_path.joinpath('offsides.csv.xz'))

    # Combine TWOSIDES PRR files and save to a single table file
    combine_prr_files(data_path.joinpath('prr/2/'),
                      tables_path.joinpath('twosides.csv.xz'))

    # shutil.rmtree('/data/prr/')

    # Save OFFSIDES propensity score files
    combine_files_to_archive(
        list(data_path.glob('scores/1/*.npz')),
        output_archive_path.joinpath('offsides_propensity_scores.tar.xz')
    )

    # Save TWOSIDES propensity score files
    # combine_files_to_archive(
    #     list(data_path.glob('scores/2/*.csv.xz')),
    #     output_archive_path.joinpath('twosides_propensity_scores.tar.xz')
    # )

    # shutil.rmtree('/data/scores/')

    # shutil.rmtree('/data/archives/')


if __name__ == "__main__":
    main()
