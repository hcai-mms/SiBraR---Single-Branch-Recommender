from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from algorithms.base_classes import SGDBasedRecommenderAlgorithm, RecommenderAlgorithm
from data.config_classes import FeatureType, EvalConfig
from data.dataset import FullEvalDataset, RecDataset
from utilities.utils import log_info_results

from natsort import natsorted
from rmet import (UserFeature, calculate, calculate_for_feature,
                  supported_metrics, supported_user_metrics, supported_distribution_metrics)
from pathlib import Path
import dill as pickle


class FullEvaluator:
    """
    Helper class for the evaluation. It considers grouping at the level of the users. When called with eval_batch, it
    updates the internal results. After the last batch, get_results will return the metrics. It holds a special group
    with index -1 that is the "ALL" user group.
    """

    def __init__(self, config: EvalConfig, evaluator_name: str = None, dataset: RecDataset = None):
        """
        :param config: Config that defines which metrics should be computed
        :param evaluator_name (optional): Name of evaluator to differentiate between results on, e.g., validation
                                          and test set. If specified, this will lead to prepending '{evaluator_name}/'
                                          to all the metrics in the results dictionary.
        :param dataset (optional): The dataset for which recommendations should be made. If supplied, metrics for the
                                   different user groups (user features of type category) are calculated.
        """
        self.config = config
        self.name = evaluator_name
        self.dataset = dataset

        # ensure that only valid metrics are supplied
        invalid_metrics = set(self.config.metrics) - set(supported_metrics)
        if len(invalid_metrics) > 0:
            raise ValueError(f'Metric(s) {invalid_metrics} are not supported. Select metrics from {supported_metrics}.')

        # determine to which class the different metrics belong to
        self._user_metrics = set(self.config.metrics).intersection(supported_user_metrics)
        self._dist_metrics = set(self.config.metrics).intersection(supported_distribution_metrics)

        # distribution metrics require access to the top_k of all users under consideration
        self._store_top_k = len(self._dist_metrics) > 0
        self._top_k = None

        # determine all user features for which we should compute group-wise metrics
        self._user_features = self._determine_user_features()

        # internal storage for the results
        self._metric_results = None
        self._distr_metric_results = None
        self._reset_internal_dict()

    def _reset_internal_dict(self):
        self._metric_results = defaultdict(lambda: list())
        self._distr_metric_results = dict()
        self._top_k = list()

    def _extend_results_dict(self, results: dict[str, float | np.ndarray | torch.Tensor]):
        for k, v in results.items():
            if isinstance(v, float):
                self._distr_metric_results[k] = v
            else:
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().numpy()
                self._metric_results[k].append(v)

    def _determine_user_features(self):
        if self.config.calculate_group_metrics:
            if self.config.user_group_features is not None:
                features = self.config.user_group_features
                for feature_name in features:
                    if feature_name not in self.dataset.user_feature_names:
                        raise ValueError(f'Dataset does not contain user feature "{feature_name}". '
                                         f'Check config whether features are loaded or set to "None" to use'
                                         f'all available categorical features.')

                    # only categorical features are supported
                    if not self.dataset.user_features[feature_name].feature_definition.type == FeatureType.CATEGORICAL:
                        raise ValueError(f'User feature "{feature_name}" is not categorical.')
            else:
                features = [ufd.name for ufd in self.dataset.user_feature_definitions
                            if ufd.type == FeatureType.CATEGORICAL]
            return features
        return None

    @staticmethod
    def _filter_rename_individual(d: dict):
        return {k.replace('_individual', ''): v for k, v in d.items() if '_individual' in k}

    def _calculate_overall_metrics(self, logits: torch.Tensor, y_true: torch.Tensor):
        results, top_k = calculate(metrics=self._user_metrics, logits=logits, targets=y_true, k=self.config.top_k,
                                   return_aggregated=True, return_individual=True, flatten_results=True,
                                   flattened_results_prefix=self.name, n_items=self.dataset.n_items_in_split,
                                   return_best_logit_indices=True)
        self._extend_results_dict(self._filter_rename_individual(results))
        return top_k.detach().cpu()

    def _calculate_group_metrics(self, u_idxs: torch.Tensor, logits: torch.Tensor, y_true: torch.Tensor):
        if self.config.calculate_group_metrics:
            # calculate results for the different features
            for feature_name in self._user_features:
                user_feature = self.dataset.user_features[feature_name]
                feature_values = user_feature[u_idxs.detach().cpu().numpy()].flatten()
                feature_labels = [lbl.lower() if isinstance(lbl, str) else lbl
                                  for lbl in user_feature.get_labels(feature_values)]
                group = UserFeature(name=feature_name, labels=feature_labels)
                feature_results = calculate_for_feature(group=group, metrics=self._user_metrics, logits=logits,
                                                        targets=y_true, k=self.config.top_k, return_individual=True,
                                                        flatten_results=True,
                                                        flattened_results_prefix=self.name)
                self._extend_results_dict(self._filter_rename_individual(feature_results))

    def eval_batch(self, u_idxs: torch.Tensor, logits: torch.Tensor, y_true: torch.Tensor):
        """
        :param u_idxs: User indexes. Shape is (batch_size).
        :param logits: Logits. Shape is (batch_size, n_items).
        :param y_true: the true prediction. Shape is (batch_size, n_items)
        """
        if logits.shape != y_true.shape:
            raise AttributeError(f'logits and true labels must have the same shape ({logits.shape} != {y_true.shape})')
        if len(u_idxs) != len(logits) or len(u_idxs) != len(y_true):
            raise AttributeError(f'assumed batch size is not equal for user indices, logits and true labels'
                                 f'({len(u_idxs)}, {len(logits)} and {len(y_true)}, respectively)')

        top_k = self._calculate_overall_metrics(logits, y_true)
        self._calculate_group_metrics(u_idxs, logits, y_true)

        if self._store_top_k:
            self._top_k.append(top_k)

    def _calculate_distribution_metrics(self):
        top_k = torch.cat(self._top_k)
        results = calculate(metrics=self._dist_metrics, k=self.config.top_k,
                            return_aggregated=True, return_individual=True, flatten_results=True,
                            flattened_results_prefix=self.name,
                            best_logit_indices=top_k, n_items=self.dataset.n_items_in_split)
        self._extend_results_dict(results)

    def get_results(self, return_raw_results: bool = False):
        metrics_dict, raw_results = {}, {}
        if len(self._user_metrics) > 0:
            # aggregate user-level metrics
            raw_results = {k: np.concatenate(v) for k, v in self._metric_results.items()}
            metrics_dict = {k: v.mean().item() for k, v in raw_results.items()}
            if self.config.calculate_std:
                metrics_dict.update({f'{k}_std': v.std().item() for k, v in raw_results.items()})

        # calculate distribution metrics
        if len(self._dist_metrics) > 0:
            self._calculate_distribution_metrics()
            metrics_dict.update(self._distr_metric_results)

        # sort keys of dictionary
        metrics_dict = {k: metrics_dict[k] for k in natsorted(metrics_dict.keys())}

        self._reset_internal_dict()

        if return_raw_results:
            return metrics_dict, raw_results
        return metrics_dict


def evaluate_recommender_algorithm(alg: RecommenderAlgorithm, eval_loader: DataLoader, evaluator: FullEvaluator,
                                   device='cpu', return_raw=False, verbose=False):
    """
    Evaluation procedure that calls FullEvaluator on the dataset.
    """
    dataset = eval_loader.dataset
    if not isinstance(dataset, FullEvalDataset):
        # we rely on this fact later
        raise ValueError("Dataset underlying loader must be of type 'FullEvaluatorDataset'")

    desc = 'evaluating' + (f' {evaluator.name}' if evaluator.name else '')
    if verbose:
        iterator = tqdm(eval_loader, desc=desc)
    else:
        print(f'{desc}...')
        iterator = eval_loader

    if not isinstance(alg, SGDBasedRecommenderAlgorithm):
        for u_idxs, i_idxs, labels in iterator:
            u_idxs = u_idxs.to(device)
            i_idxs = i_idxs.to(device)
            labels = labels.to(device)

            out = alg.predict(u_idxs, i_idxs)

            batch_mask = torch.tensor(dataset.exclude_data[u_idxs.cpu().numpy()].toarray(), dtype=torch.bool)
            out[batch_mask] = -torch.inf

            if not isinstance(out, torch.Tensor):
                out = torch.tensor(out).to(device)

            evaluator.eval_batch(u_idxs, out, labels)
    else:
        # prevent model from being evaluated in train mode
        alg.eval()

        with torch.no_grad():
            # We generate the item representation once (usually the bottleneck of evaluation)
            i_idxs = torch.tensor(dataset.items_in_split).to(device)
            i_repr = alg.get_item_representations(i_idxs)

            for u_idxs, _, labels in iterator:
                u_idxs = u_idxs.to(device)
                labels = labels.to(device)

                u_repr = alg.get_user_representations(u_idxs)
                out = alg.combine_user_item_representations(u_repr, i_repr)

                batch_mask = torch.tensor(dataset.exclude_data[u_idxs.cpu()].toarray(), dtype=torch.bool)
                out[batch_mask] = -torch.inf

                evaluator.eval_batch(u_idxs, out, labels)

    results = evaluator.get_results(return_raw_results=return_raw)
    # select only metrics for logging (in case raw results are also returned)
    log_info_results(results[0] if isinstance(results, tuple) else results)
    return results


class Gatherer:
    def __init__(self):
        self._obj_collection = None
        self._collection = None
        self.reset()

    def reset(self):
        self._obj_collection = dict()
        self._collection = defaultdict(lambda: list())

    def add(self, name: str, values: any):
        if isinstance(values, (np.ndarray, torch.Tensor)):
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().numpy()
            self._collection[name].append(values)
        else:
            self._obj_collection[name] = values

    def gather(self):
        results = {k: np.concatenate(v) for k, v in self._collection.items()}
        results.update(self._obj_collection)
        return results

    def export_pkl(self, path: str):
        with open(path, 'wb') as fh:
            pickle.dump(self.gather(), fh)


def gather_recommender_algorithm_results(alg: RecommenderAlgorithm, eval_loader: DataLoader, evaluator: FullEvaluator,
                                         results_path: str = None, device: str = 'cpu', verbose: bool = False):
    """
    Gathers the logits of a recommender system on the specified data loader
    """
    dataset = eval_loader.dataset
    if not isinstance(dataset, FullEvalDataset):
        # we rely on this fact later
        raise ValueError("Dataset underlying loader must be of type 'FullEvaluatorDataset'")

    if verbose:
        iterator = tqdm(eval_loader, desc='gathering data')
    else:
        iterator = eval_loader

    k = max(evaluator.config.top_k)

    gatherer = Gatherer()
    gatherer.add('n_users', dataset.n_users_in_split)
    gatherer.add('n_items', dataset.n_items_in_split)
    gatherer.add('k', k)

    if not isinstance(alg, SGDBasedRecommenderAlgorithm):
        # as FullEvalDataset is used, each item in the batch consists of
        # ({user index}, {item indices in split}, {labels for all item indices})
        for u_idxs, i_idxs, labels in iterator:
            u_idxs = u_idxs.to(device)
            i_idxs = i_idxs.to(device)
            labels = labels.to(device)

            out = alg.predict(u_idxs, i_idxs)

            batch_mask = torch.tensor(dataset.exclude_data[u_idxs.cpu()].toarray(), dtype=torch.bool)
            out[batch_mask] = -torch.inf

            if not isinstance(out, torch.Tensor):
                out = torch.tensor(out).to(device)
            evaluator.eval_batch(u_idxs, out, labels)

            top_entries = torch.topk(out, k, largest=True, sorted=True)
            gatherer.add('topk_item_indices', top_entries.indices)
            gatherer.add('topk_logits', top_entries.values)
            gatherer.add('user_indices', u_idxs)
            gatherer.add('targets', torch.argwhere(labels))
    else:
        alg.eval()
        with torch.no_grad():
            # We generate the item representation once (usually the bottleneck of evaluation)
            i_idxs = torch.tensor(dataset.items_in_split).to(device)
            i_repr = alg.get_item_representations(i_idxs)

            for u_idxs, _, labels in iterator:
                u_idxs = u_idxs.to(device)
                labels = labels.to(device)

                u_repr = alg.get_user_representations(u_idxs)
                out = alg.combine_user_item_representations(u_repr, i_repr)

                batch_mask = torch.tensor(dataset.exclude_data[u_idxs.cpu()].toarray(), dtype=torch.bool)
                out[batch_mask] = -torch.inf
                evaluator.eval_batch(u_idxs, out, labels)

                top_entries = torch.topk(out, k, largest=True, sorted=True)
                gatherer.add('topk_item_indices', top_entries.indices)
                gatherer.add('topk_logits', top_entries.values)
                gatherer.add('user_indices', u_idxs)
                gatherer.add('targets', torch.argwhere(labels))

    metrics, raw_metrics = evaluator.get_results(return_raw_results=True)
    gatherer.add('metrics', metrics)
    gatherer.add('raw_metrics', raw_metrics)

    if results_path is not None:
        gatherer.export_pkl(results_path)

    return gatherer.gather()
