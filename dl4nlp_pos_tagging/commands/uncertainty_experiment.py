
import argparse
from copy import deepcopy
import inspect
import logging
import os
from os import PathLike
from typing import Any, Dict, List, Optional, Union, Tuple

from allennlp.common import Params
from allennlp.commands.subcommand import Subcommand
from allennlp.commands.train import train_model
from overrides import overrides

import dl4nlp_pos_tagging.common.utils as utils
from dl4nlp_pos_tagging.config import Config
from dl4nlp_pos_tagging.experiments.uncertainty_experiment import UncertaintyExperiment

logger = logging.getLogger(Config.logger_name)

@Subcommand.register("uncertainty-experiment")
class RunUncertaintyExperiment(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = (\
            "Train a morphosyntactic tagger and calculate the uncertainty of its POS tag predictions. "\
            "All results are saved in the specified output directory."\
        )

        subparser = parser.add_parser(self.name, description=description)

        subparser.add_argument(
            "experiment_path",
            type=str,
            help="path to the .jsonnet file"
        )

        subparser.add_argument(
            "-o",
            "--serialization-dir",
            type=str,
            default=None,
            help="base directory in which to save the model and its artifacts"
        )

        subparser.add_argument(
            "-r",
            "--recover",
            action="store_true",
            default=False,
            help="recover training from the state in serialization_dir."
        )

        subparser.add_argument(
            "-f",
            "--force",
            action="store_true",
            required=False,
            help="overwrite the output directory, including ALL trained models and dataframes if it exists."
        )

        subparser.add_argument(
            "--train-only",
            action="store_true",
            required=False,
            help="Only train the model, do not calculate uncertainty."
        )

        subparser.set_defaults(func=run_uncertainty_experiment_from_args)

        return subparser

def run_uncertainty_experiment_from_args(args: argparse.Namespace):
    """
    Just converts from an `argparse.Namespace` object to string paths.
    """
    run_uncertainty_experiment_from_file(
        experiment_filename=args.experiment_path,
        serialization_dir=args.serialization_dir,
        recover=args.recover,
        force=args.force,
        train_only=args.train_only
    )

def run_uncertainty_experiment_from_file(
    experiment_filename: PathLike,
    serialization_dir: Optional[Union[str, PathLike]] = None,
    recover: Optional[bool] = False,
    force: Optional[bool] = False,
    train_only: Optional[bool] = False
):
    """
    A wrapper around `run_uncertainty_experiment` which loads the params from a file.
    """
    experiment_name = os.path.splitext(os.path.basename(experiment_filename))[0]

    if not serialization_dir:
        serialization_dir = os.path.join(Config.serialization_base_dir, experiment_name)

    params = Params.from_file(experiment_filename)

    run_uncertainty_experiment(
        params=params,
        name=experiment_name,
        serialization_dir=serialization_dir,
        recover=recover,
        force=force,
        train_only=train_only
    )

def run_uncertainty_experiment(
    params: Params,
    name: str,
    serialization_dir: PathLike,
    recover: Optional[bool] = False,
    force: Optional[bool] = False,
    train_only: Optional[bool]= False
):

    train_params = deepcopy(params)

    if "uncertainty_experiment" not in train_params.params:
        logger.warn("Configuration file missing 'uncertainty_experiment' parameters. Enabling train-only mode.")
        train_only = True
        experiment_params = None
    else:
        experiment_params = train_params.params.pop("uncertainty_experiment")
        experiment_params['test_data_path'] = train_params['test_data_path']
        experiment_params = Params(experiment_params)

    should_train = any([force, not utils.model_already_trained(serialization_dir), recover])

    if should_train:
        train_model(
                params=train_params,
                serialization_dir=serialization_dir,
                recover=recover,
                force=force
            )

    if train_only:
        logger.info("'train-only' was specified. Finishing without calculating uncertainty.")
        return

    uncertainty_experiment = UncertaintyExperiment.from_params(
        params=experiment_params,
        serialization_dir=serialization_dir,
    )
    uncertainty_experiment.calculate_uncertainty(force=force)
    uncertainty_experiment.generate_artifacts()
