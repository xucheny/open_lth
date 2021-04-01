import dataclasses
import numpy as np

from foundations import hparams
import models.base
from pruning import base
from pruning.mask import Mask


@dataclasses.dataclass
class PruningHparams(hparams.PruningHparams):
    pruning_fraction: float = 0.2
    pruning_layers_to_ignore: str = None

    _name = 'Hyperparameters for VQE Sparse Global Pruning in VQE'
    _description = 'Hyperparameters that modify the way pruning occurs.'
    _pruning_fraction = 'The fraction of additional weights to prune from the network.'
    _layers_to_ignore = 'A comma-separated list of addititonal tensors that should not be pruned.'


class Strategy(base.Strategy):
    @staticmethod
    def get_pruning_hparams() -> type:
        return PruningHparams

    @staticmethod
    def prune(pruning_hparams: PruningHparams, trained_model: models.base.Model, current_mask: Mask = None):
        current_mask = Mask.ones_like(trained_model).numpy() if current_mask is None else current_mask.numpy()
        # number of initializations
        num_inits = next(iter(current_mask.values())).shape[0]
        assert np.array([num_inits == v.shape[0] for v in current_mask.values()]).all()

        # Determine the number of weights that need to be pruned.
        number_of_remaining_weights_per_init = np.sum([np.sum(v) for v in current_mask.values()]) // num_inits
        number_of_weights_to_prune_per_init = np.ceil(
            pruning_hparams.pruning_fraction * number_of_remaining_weights_per_init).astype(int)

        # Determine which layers can be pruned.
        prunable_tensors = set(trained_model.prunable_layer_names)
        if pruning_hparams.pruning_layers_to_ignore:
            prunable_tensors -= set(pruning_hparams.pruning_layers_to_ignore.split(','))

        # Get the model weights.
        weights = {k: v.clone().cpu().detach().numpy()
                   for k, v in trained_model.state_dict().items()
                   if k in prunable_tensors}

        # Create a vector of all the unpruned weights in the model.
        weight_vectors = [
                np.concatenate(
                    [
                        v[init_id, ...][current_mask[k][init_id,...] == 1]
                        for k, v in weights.items()
                        ]
                    )
                for init_id in range(num_inits)]
        thresholds = np.array([
                np.sort(np.abs(wv))[number_of_weights_to_prune_per_init] for wv in weight_vectors
                ])
        mask_dict = {}
        for k, v in weights.items():
            threshold_tensor = thresholds.reshape(-1, *[1 for _ in range(v.ndim-1)])
            threshold_tensor = np.tile(threshold_tensor, v.shape[1:])
            mask_dict[k] = np.where(np.abs(v) > threshold_tensor, current_mask[k], np.zeros_like(v))
        new_mask = Mask(mask_dict)
        for k in current_mask:
            if k not in new_mask:
                new_mask[k] = current_mask[k]

        return new_mask
