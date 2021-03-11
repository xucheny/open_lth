# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import torch

from datasets.base import DataLoader
from foundations import hparams
from foundations.step import Step
from platforms.platform import get_platform
from training import checkpointing


# Standard callbacks.
def save_model(output_location, step, model, optimizer, logger):
    model.save(output_location, step)


def save_logger(output_location, step, model, optimizer, logger):
    logger.save(output_location)


def create_timekeeper_callback():
    time_of_last_call = None

    def callback(output_location, step, model, optimizer, logger):
        if get_platform().is_primary_process:
            nonlocal time_of_last_call
            t = 0.0 if time_of_last_call is None else time.time() - time_of_last_call
            print(f'Ep {step.ep}\tIt {step.it}\tTime Elapsed {t:.2f}')
            time_of_last_call = time.time()
        get_platform().barrier()

    return callback


def create_loss_eval_callback(verbose=False):
    """This function returns a callback."""

    time_of_last_call = None

    def eval_callback(output_location, step, model, optimizer, logger):
        model.eval()

        with torch.no_grad():
            output = model(None)
            energy_gap = model.loss_criterion(output, None)

        # Share the information if distributed.
        if get_platform().is_distributed:
            raise NotImplementedError('distribution evaluation for lossonly callback not implemented')
            torch.distributed.reduce(energy_gap, 0, op=torch.distributed.ReduceOp.SUM)

        energy_gap = energy_gap.cpu().item()

        if get_platform().is_primary_process:
            logger.add('energy_gap', step, energy_gap)

            if verbose:
                nonlocal time_of_last_call
                elapsed = 0 if time_of_last_call is None else time.time() - time_of_last_call
                print('ep {:03d}\tit {:03d}\tgap {:.5f}\ttime {:.2f}s'.format(
                    step.ep, step.it, energy_gap, elapsed))
                time_of_last_call = time.time()

    return eval_callback


# Callback frequencies. Each takes a callback as an argument and returns a new callback
# that runs only at the specified frequency.
def run_every_epoch(callback):
    def modified_callback(output_location, step, model, optimizer, logger):
        if step.it != 0:
            return
        callback(output_location, step, model, optimizer, logger)
    return modified_callback


def run_every_step(callback):
    return callback


def run_at_step(step1, callback):
    def modified_callback(output_location, step, model, optimizer, logger):
        if step != step1:
            return
        callback(output_location, step, model, optimizer, logger)
    return modified_callback


# The set of callbacks that should be used when we don't care about the dataset.
def lossonly_callbacks(training_hparams: hparams.TrainingHparams, train_set_loader: DataLoader,
                       test_set_loader: DataLoader, eval_on_train: bool = False, verbose: bool = True,
                       start_step: Step = None, evaluate_every_epoch: bool = True):
    start = start_step or Step.zero(train_set_loader.iterations_per_epoch)
    end = Step.from_str(training_hparams.training_steps, train_set_loader.iterations_per_epoch)
    loss_eval_callback = create_loss_eval_callback(verbose=verbose)

    # Basic checkpointing and state saving at the beginning and end.
    result = [
        run_at_step(start, save_model),
        run_at_step(end, save_model),
        run_at_step(end, save_logger),
        run_every_epoch(checkpointing.save_checkpoint_callback),
    ]

    # Test every epoch if requested.
    if evaluate_every_epoch: result = [run_every_epoch(loss_eval_callback)] + result
    elif verbose: result.append(run_every_epoch(create_timekeeper_callback()))

    # Ensure that testing occurs at least at the beginning and end of training.
    if start.it != 0 or not evaluate_every_epoch: result = [run_at_step(start, loss_eval_callback)] + result
    if end.it != 0 or not evaluate_every_epoch: result = [run_at_step(end, loss_eval_callback)] + result

    return result
