from typing import List, Dict, Tuple, Union, Optional
import time
import sys
import numpy as np
import json
import os
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import argparse

from humanprompt.tasks.dataset_loader import DatasetLoader
from humanprompt.utils.config_utils import load_config


def extract_embeddings(dataset, embed_model, task_type) -> Dict[int, np.array]:
    """
    Extract embeddings from a dataset using a model.

    Args:
        dataset: Dataset to extract embeddings from.
        embed_model: Model to extract embeddings with.
    """
    embed_dict = {}
    for idx, data_item in enumerate(dataset):
        if task_type == "simple qa":
            embeddings = embed_model.encode(data_item["question"])
        elif task_type == "multi-choice qa":
            text = data_item["question"] + "; ".join(data_item["choices"]["text"])
            embeddings = embed_model.encode(text)
        elif task_type == "nli":
            text = data_item["hypothesis"] + " " + data_item["premise"]
            embeddings = embed_model.encode(text)
        else:
            raise ValueError(f"Unknown task type {task_type}")
        embed_dict[idx] = embeddings
    return embed_dict


def _group_samples_clustering(embed_dict: Dict[int, np.array]) -> List[List[int]]:
    def _calculate_cos_similarities(v1: np.array, v2: np.array):
        num = np.dot(v1, v2.T)
        denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)
        similarity_matrix = num / denom
        similarity_matrix[np.isneginf(similarity_matrix)] = 0
        similarity_matrix = 0.5 + 0.5 * similarity_matrix
        return similarity_matrix

    if len(embed_dict) % num_in_batch:
        n_clusters = len(embed_dict) // num_in_batch + 1
    else:
        n_clusters = len(embed_dict) // num_in_batch
    # K-means clustering
    embed_matrix = np.array(list(embed_dict.values()))  # [n_samples, h]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(embed_matrix)
    similarity_matrix = _calculate_cos_similarities(embed_matrix, kmeans.cluster_centers_)  # [n_samples, n_clusters]
    similarity_rankings = np.argsort(-similarity_matrix, axis=1)  # [n_samples, n_clusters]
    groups = [[] for _ in range(n_clusters)]
    for sample_idx, label in enumerate(kmeans.labels_):
        groups[label].append(sample_idx)
    # Reassign to equalize the number of samples in each cluster
    for group_idx, group in enumerate(groups):
        if len(group) > num_in_batch:
            groups[group_idx] = sorted(group, key=lambda x: similarity_matrix[x, group_idx], reverse=True)
            samples_to_reassign = groups[group_idx][num_in_batch:]
            groups[group_idx] = groups[group_idx][:num_in_batch]
            for sample_idx in samples_to_reassign:
                for candi_group_idx in similarity_rankings[sample_idx]:
                    if len(groups[candi_group_idx]) < num_in_batch:
                        groups[candi_group_idx].append(sample_idx)
                        break
    return groups


def _group_samples_diversity(embed_dict: Dict[int, np.array]) -> List[List[int]]:
    def fast_votek(embeddings, select_num, k, vote_file=None):
        """
        vote-k algorithm that selects the most diverse K samples.
        Ref "https://github.com/HKUNLP/icl-selective-annotation/blob/main/two_steps.py"
        """
        n = len(embeddings)
        if vote_file is not None and os.path.isfile(vote_file):
            with open(vote_file) as f:
                vote_stat = json.load(f)
        else:
            vote_stat = defaultdict(list)
            for i in range(n):
                cur_emb = embeddings[i].reshape(1, -1)
                cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
                sorted_indices = np.argsort(cur_scores).tolist()[-k - 1:-1]
                for idx in sorted_indices:
                    if idx != i:
                        vote_stat[idx].append(i)
            if vote_file is not None:
                with open(vote_file, 'w') as f:
                    json.dump(vote_stat, f)
        votes = sorted(vote_stat.items(), key=lambda x: len(x[1]), reverse=True)
        selected_indices = []
        selected_times = defaultdict(int)
        while len(selected_indices) < select_num:
            cur_scores = defaultdict(int)
            for idx, candidates in votes:
                if idx in selected_indices:
                    cur_scores[idx] = -100
                    continue
                for one_support in candidates:
                    if not one_support in selected_indices:
                        cur_scores[idx] += 10 ** (-selected_times[one_support])
            cur_selected_idx = max(cur_scores.items(), key=lambda x: x[1])[0]
            selected_indices.append(int(cur_selected_idx))
            for idx_support in vote_stat[cur_selected_idx]:
                selected_times[idx_support] += 1
        return selected_indices

    embeddings = list(embed_dict.values())
    selected_indices = fast_votek(
        embeddings=embeddings,
        select_num=len(embeddings),
        k=len(embeddings) // 5,
        vote_file=votek_embedding_file
    )
    groups, one_group_indices = [], []
    for idx, selected_indice in enumerate(selected_indices):
        one_group_indices.append(selected_indice)
        if len(one_group_indices) == num_in_batch \
                or idx == len(selected_indices) - 1:
            groups.append(one_group_indices)
            one_group_indices = []
    print(len(groups), groups)
    return groups


def group_samples(embed_dict: Dict[int, np.array], group_method: str) -> List[List[int]]:
    """
    Group samples by clustering

    Args:
        embed_dict: Embeddings to group.
    """
    if group_method == "clustering":
        return _group_samples_clustering(embed_dict)
    elif group_method == "diversity":
        return _group_samples_diversity(embed_dict)
    else:
        raise ValueError(f"Unknown group method {group_method}")


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="batch_inference-gsm8k", help="Experiment name.")
    parser.add_argument("--task_type", type=str, default="simple qa",
                        choices=["simple qa", "multi-choice qa", "nli"], help="Task type, deciding the input format.")
    parser.add_argument("--embed_model_name", type=str, default="sentence-transformers/paraphrase-mpnet-base-v2",
                        help="The model name of the sentence embedding model.")
    parser.add_argument("--group_method", type=str, default="diversity",
                        choices=["diversity", "clustering"], help="Group method.")
    parser.add_argument("--num_in_batch", type=int, default=4, help="Number of samples in one batch.")
    parser.add_argument("--num_test_samples", type=int, default=300,
                        help="Number of test samples. Set None to use all.")
    parser.add_argument("--save_dir", type=str, default="group_results/", help="Directory to save grouping results.")
    args = parser.parse_args()

    # Meta-config
    exp_name = args.exp_name
    group_method = args.group_method
    num_in_batch = args.num_in_batch
    exp_config = load_config(f"../configs/{exp_name}-batch={num_in_batch}.yaml")
    dataset_name = exp_name.split('-')[1]
    votek_embedding_file = f".cache/{dataset_name}-batch={num_in_batch}-votek_cache.json"
    task_type = args.task_type
    embed_model_name = args.embed_model_name
    num_test_samples = args.num_test_samples
    group_exp_name = f"{dataset_name}-batch={num_in_batch}-{group_method}"
    save_dir = args.save_dir
    save_file = f"{save_dir}/{group_exp_name}.json"

    # Config
    if not hasattr(exp_config, "dataset"):
        raise ValueError("Experiment config must have a `dataset` field.")

    dataset_config = exp_config["dataset"]
    dataset = DatasetLoader.load_dataset(
        dataset_name=dataset_config["dataset_name"],
        dataset_split=dataset_config["dataset_split"],
        dataset_subset_name=dataset_config["dataset_subset_name"]
        if "dataset_subset_name" in dataset_config else None,
        dataset_key_map=dataset_config["dataset_key_map"]
        if "dataset_key_map" in dataset_config else None,
    )
    if num_test_samples:
        dataset = dataset.select(range(num_test_samples))

    # Embed
    start_time = time.time()
    embed_model = SentenceTransformer(embed_model_name)
    embed_dict = extract_embeddings(dataset, embed_model, task_type)
    print(f"Embedding extraction took {time.time() - start_time} seconds.")

    # Group samples
    start_time = time.time()
    groups = group_samples(embed_dict, group_method)
    print(f"Grouping samples took {time.time() - start_time} seconds.")
    with open(save_file, "w") as f:
        json.dump(groups, f)
