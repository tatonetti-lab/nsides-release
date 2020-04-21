import tarfile

import numpy

import parallel_utils
import utils


class PRRComputer:
    def __init__(self, exposures, outcomes, drug_id_vector, outcome_id_vector):
        self.exposures = exposures
        self.outcomes = outcomes
        self.drug_id_vector = drug_id_vector
        self.outcome_id_vector = outcome_id_vector
        self.n_reports = self.exposures.shape[0]

    def compute_prr_from_scores_file(self, propensity_scores_path):
        file_info = utils.extract_filepath_info(propensity_scores_path.name)
        drug_indices = file_info[1:]
        propensity_scores = numpy.load(propensity_scores_path)[:self.n_reports]
        return self.compute_prr(drug_indices, propensity_scores)

    def compute_prr(self, drug_indices, propensity_scores,
                    bins=numpy.arange(0, 1.2, 0.2), seed=0):
        drug_exposures = utils.compute_multi_exposure(drug_indices,
                                                      self.exposures)
        prr_df = parallel_utils._prr_helper(
            scores=propensity_scores,
            drug_exposures=drug_exposures,
            all_outcomes=self.outcomes,
            outcome_id_vector=self.outcome_id_vector)
        return prr_df


class Logger:
    def __init__(self, success_log_path, failure_log_path):
        self.success = success_log_path
        self.failure = failure_log_path

        open(self.success, 'w+').close()
        open(self.failure, 'w+').close()

    def log_error(self, path):
        with open(self.failure, 'a') as f:
            f.write(f"{path}\n")

    def log_success(self, path):
        with open(self.success, 'a') as f:
            f.write(f"{path}\n")


def unpack_archive(path, output_path):
    try:
        tar = tarfile.open(path, mode='r:gz')
        members = tar.getmembers()
    except:  # noqa:E722
        return None

    scores_members = [member for member in members if 'score' in member.name]
    tar.extractall(path=output_path, members=scores_members)

    return [
        output_path.joinpath(member.name) for member in scores_members
    ]
