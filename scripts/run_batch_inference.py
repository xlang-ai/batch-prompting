import os
import ssl
from typing import Dict, List, Union
import json
from datasets import Dataset
import openai
import time
import argparse
import os
import ssl
import sys
from typing import Dict, List, Union
import json
from datasets import Dataset
import openai
import time
import argparse
sys.path.insert(1,r"C:\Users\alexs\PycharmProjects\batch-prompting")  # Add the project root to the path
from hub.cot.commonsense_qa.extract_cot_commonsense_qa import CoTCommonsenseQAExtract
from humanprompt.evaluators.evaluator import Evaluator
from humanprompt.methods.auto.method_auto import AutoMethod
from humanprompt.methods.base_method.method import PromptMethod
from humanprompt.tasks.dataset_loader import DatasetLoader
from humanprompt.utils.config_utils import load_config

from humanprompt.evaluators.evaluator import Evaluator
from humanprompt.methods.auto.method_auto import AutoMethod
from humanprompt.methods.base_method.method import PromptMethod
from humanprompt.tasks.dataset_loader import DatasetLoader
from humanprompt.utils.config_utils import load_config

# ADDITIONAL IMPORT FOR INTER-ARRIVAL TIME
import random # to simulate randomness
#########################################



class OpenAIKeyPool:
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.idx = 0

    def get_key(self):
        key = self.keys[self.idx]
        self.idx += 1
        if self.idx == len(self.keys):
            self.idx = 0
        return key


def run_experiment(
    dataset: Dataset,
    method: PromptMethod,
    evaluator: Evaluator,
) -> Dict:
    # NEW DATA FOR INTER-ARRIVAL
    arrival_times = [] # List to store arrival timestamps

    """
    Run experiment on a dataset using a method.

    Args:
        dataset: Dataset to run experiment on.
        method: Method to run experiment with.
        evaluator: Evaluator to evaluate the experiment performance.
    """
    batch_data_items = []
    predictions, gold_answers = [], []
    for idx, data_item in enumerate(dataset):
        data_item['idx'] = idx

        # ADDITIONAL CODE FOR RANDOM INTER-ARRIVAL TIME
        # Generate a random inter-arrival time (e.g., uniformly distributed between 1 and 5 seconds)
        random_interarrival = random.uniform(1,5)
        time.sleep(random_interarrival) # get a random waiting period

        arrival_time = time.time() # Record the arrival timestamp
        arrival_times.append(arrival_time) # Append the arrival time to our list of arrival times

        ################################################

        if data_item.get('id', None) is None:
            data_item['id'] = idx
        if use_cache \
                and os.path.exists(os.path.join(tmp_save_dir, f"{idx}_{data_item['id']}.json")):
            # Already inferenced example
            with open(
                os.path.join(tmp_save_dir, f"{idx}_{data_item['id']}.json"), "r"
            ) as f:
                result_item = json.load(f)
                batch_prediction, batch_gold_answer = (
                    [result_item["prediction"]],
                    [result_item["gold_answer"]],
                )
                print(f"idx: {data_item['idx']}")
                print(f"id: {data_item['id']}")
                print(f"pred answer: {result_item['prediction']}")
                print(f"gold answer: {result_item['gold_answer']}")
        else:
            # New coming example
            batch_data_items.append(data_item)
            if len(batch_data_items) < num_in_batch \
                    and idx != len(dataset) - 1:
                continue
            while True:
                try:
                    current_key = openai_key_pool.get_key()
                    os.environ["OPENAI_API_KEY"] = current_key
                    start_time = time.time()
                    print("Using OpenAI key: ", current_key)
                    batch_prediction = method.run(
                        x=batch_data_items,
                        verbose=verbose
                    )
                    print("One inference time: ", time.time() - start_time)
                    break
                except openai.OpenAIError as e:
                    print(f"Error when getting response: {e}")
                    continue
            # Clean batch prediction
            if idx != len(dataset) - 1:
                target_num_in_batch = num_in_batch
            else:
                # The last batch
                num_in_last_batch = len(dataset) % num_in_batch if len(dataset) % num_in_batch != 0 else num_in_batch
                target_num_in_batch = num_in_last_batch
            if batch_prediction is None:
                batch_prediction = ["<empty>"] * target_num_in_batch
            if len(batch_prediction) < target_num_in_batch:
                batch_prediction = batch_prediction + ["<empty>"] * (target_num_in_batch - len(batch_prediction))
            elif len(batch_prediction) > target_num_in_batch:
                batch_prediction = batch_prediction[:target_num_in_batch]
            # Answer post-processing for evaluation
            batch_prediction = evaluator.normalize_answer(batch_prediction)
            batch_gold_answer = evaluator.normalize_answer([data_item['answer'] for data_item in batch_data_items])
            print(len(batch_prediction), len(batch_gold_answer))
            # Cache current example
            for data_item, prediction, gold_answer in \
                    zip(batch_data_items, batch_prediction, batch_gold_answer):
                os.makedirs(tmp_save_dir, exist_ok=True)
                with open(os.path.join(tmp_save_dir, f"{data_item['idx']}_{data_item['id']}.json"), "w") as f:
                    json.dump({
                        "idx": data_item['idx'],
                        "id": data_item["id"],
                        "prediction": prediction,
                        "gold_answer": gold_answer
                    }, f)
                print(f"idx: {data_item['idx']}")
                print(f"id: {data_item['id']}")
                print(f"pred answer: {prediction}")
                print(f"gold answer: {gold_answer}")
            batch_data_items = []

            #################################
            # Calculate inter-arrival time (skip for the first request)
            if idx > 0:
                inter_arrival_time = arrival_times[idx] - arrival_times[idx - 1]
                print(f"Inter-Arrival Time for request {idx}: {inter_arrival_time} seconds")

        print('-' * 80)
        predictions.extend(batch_prediction)
        gold_answers.extend(batch_gold_answer)
    # Evaluate
    print(len(predictions))
    eval_dict = evaluator.evaluate(predictions, gold_answers)
    return eval_dict


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="batch_inference-gsm8k", help="Experiment name.")
    parser.add_argument("--num_in_batch", type=int, default=2, help="Number of samples in one batch.")
    parser.add_argument("--num_test_samples", type=int, default=300,
                        help="Number of test samples. Set None to use all.")
    parser.add_argument("--debug_indices", type=str, default=None,
                        help="Debug indices of samples in dataset. Set None to use all.")
    parser.add_argument("--openai_api_key_file", type=str, default="openai_api_keys.txt", help="OpenAI API key file.")
    parser.add_argument("--save_dir", type=str, default="results/", help="Directory to save evaluation results.")
    parser.add_argument("--use_cache", type=bool, default=True,
                        help="Whether to use cache for already tested samples.")
    parser.add_argument("--verbose", action="store_true", help="Whether to print verbose information.")
    args = parser.parse_args()

    # Meta-config
    exp_name = args.exp_name
    num_in_batch = args.num_in_batch
    exp_config = load_config(f"configs/{exp_name}-batch={num_in_batch}.yaml")
    num_test_samples = args.num_test_samples
    debug_indices = args.debug_indices if args.debug_indices is None \
        else [int(x) for x in args.debug_indices.split(",")]
    with open(args.openai_api_key_file, "r") as f:
        openai_keys = f.read().splitlines()
    openai_key_pool = OpenAIKeyPool(
        keys=openai_keys
    )
    os.environ["OPENAI_API_KEY"] = openai_key_pool.get_key()
    save_dir = args.save_dir
    tmp_save_dir = os.path.join(save_dir, "tmp", f"{exp_name}/")
    use_cache = args.use_cache
    verbose = args.verbose

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
    if debug_indices:
        dataset = dataset.select(debug_indices)

    if not hasattr(exp_config, "method"):
        raise ValueError("Experiment config must have a `method` field.")

    method_config = exp_config["method"]
    method = AutoMethod.from_config(
        method_name=method_config["method_name"]
        if method_config.get("method_name") else None,
        config_file_path=method_config["method_config_file_path"]
        if method_config.get("method_config_file_path") else None,
        **method_config.get("method_args", {}),
    )
    evaluator = Evaluator(
        metrics=exp_config["metrics"],
        dataset_name=dataset_config["dataset_name"],
        dataset_subset_name=dataset_config["dataset_subset_name"]
        if "dataset_subset_name" in dataset_config
        else None,
    )

    # Run experiment
    start_time = time.time()
    eval_dict = run_experiment(dataset=dataset, method=method, evaluator=evaluator)
    print(f"Elapsed time: ", time.time() - start_time)
    print(eval_dict)
    with open(os.path.join(save_dir, f"eval_{exp_name}.json"), "w") as f:
        json.dump(eval_dict, f)
