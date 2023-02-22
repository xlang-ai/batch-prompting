# Batch Prompting
This is the official implementation of the batch prompting paper: [Batch Prompting: Efficient Inference with Large Language Model APIs](https://arxiv.org/pdf/2301.08721.pdf).
Batch prompting is a simple alternative prompting approach that enables the LLM to run inference in batches, instead of one sample at a time.
In this way, it largely saves costs of LLM API calls both computationally and financially, while achieving good performance.

This repository contains the code for benchmarking batch prompting over ten datasets in paper. The main code structure references [HKUNLP/humanprompt]((https://github.com/HKUNLP/HumanPrompt)) project, that builds LLM prompting methods as pipelines and provides a unified interface(config-and-run) for researchers to use.

**Updates**
+ Feb 2023: Release the code for benchmarking batch prompting.
+ Jan 2023: Arxiv version of the paper is available [here](https://arxiv.org/pdf/2301.08721.pdf).

## Contents
+ [Setup](#setup)
+ [Usage](#usage) 
  - [Config](#config)
  - [Run experiments](#run-experiments)
    * [OpenAI keys](#openai-keys)
    * [Run batch prompting](#run-batch-prompting)
    * [Run standard prompting](#run-standard-prompting)
    * [Run sample grouping](#run-sample-grouping)
+ [Code structure](#code-structure)
+ [Citation](#citation)

## Setup
Firstly, clone this repo, then run:
```bash
pip install -e .
```
This will install humanprompt package and add soft link hub to `./humanprompt/artifacts/hub`.

## Usage

### Config
We follow "one config, one experiment" paradigm. 
In each experiment's config file(.yaml), you can config the dataset, prompting method, and metrics.
Specifically, you can config parameters as follows (take [CommonsenseQA](https://arxiv.org/pdf/1811.00937.pdf) as an example):
```yaml
---
  dataset:
    dataset_name: "commonsense_qa"              # dataset name, aligned with huggingface dataset if loaded from it
    dataset_split: "validation"                 # dataset split
    dataset_subset_name: null                   # dataset subset name, null if not used
    dataset_key_map:                            # mapping original dataset keys to humanprompt task keys to unify the interface
      question: "question"
      choices: "choices"
      answer: "answerKey"
      id: "id"
  method:
    method_name: "batch_inference"              # method name to initialize the prompting method class
    method_config_file_path: null               # method config file path, null if not used(will be overriden by method_args).
    method_args:
      client_name: "openai"                     # LLM API client name, adopted from github.com/HazyResearch/manifest
      transform: "batch_inference.commonsense_qa.transform_batch_inference_commonsense_qa.BatchInferenceCommonsenseQATransform"  # user-defined transform class to build the prompts
      extract: "batch_inference.commonsense_qa.extract_batch_inference_commonsense_qa.BatchInferenceCommonsenseQAExtract"        # user-defined extract class to extract the answers from output
      extraction_regex: ".*So the answer is (.*).\n?"                        # user-defined regex to extract the answer from output
      prompt_file_path: "batch_inference/commonsense_qa/prompt-batch=2.txt"  # prompt file path
      max_tokens: 512                           # max generated tokens
      temperature: 0                            # temperature for generated tokens
      engine: code-davinci-002                  # LLM engine
      stop_sequence: "\n\n"                     # stop sequence for generation
  metrics:
    - "exact_match"                             # metrics to evaluate the results
```
Users can create the `transform` and `extract` classes to customize the prompt generation and answer extraction process. 
Prompt file can be replaced or specified according to the user's need.
The default model(engine) is Codex(code-davinci-002). You can change to other models by specifying the `engine` parameter.

### Run experiments
To run experiments, you can specify the experiment name and other meta configs in command line under `scripts/` directory.

#### OpenAI keys
If your client language model is "openai", you need to get your own API key(s) and save them in a file, e.g., `openai_api_key.txt`.
If multiple keys are provided(one key in one line), the prompting method will automatically switch the key to use in turn to avoid the rate limit.

#### Run batch prompting
For example, run the following command to run batch prompting on CommonsenseQA with batch size 2:
```bash
python run_batch_inference.py
  --exp_name batch_inference-commonsense_qa 
  --num_in_batch 2 
  --openai_api_key openai_api_key.txt 
  --save_dir results/ 
  --use_cache True
```

#### Run standard prompting
Similar for running standard prompting:
```bash
python run_standard.py
  --exp_name cot-commonsense_qa 
  --openai_api_key openai_api_key.txt 
  --save_dir results/ 
  --use_cache True
```

#### Run sample grouping
We experiment with "similarity" and "diversity"-based sample grouping methods in paper.
To run sample grouping, you can first run this command in `scripts/misc/` directory to get the grouping results:
```bash
python group_samples.py
  --exp_name batch_inference-commonsense_qa 
  --num_in_batch 2
  --grouping_method similarity
  --task_type "multi-choice qa"
  --embed_model_name sentence-transformers/paraphrase-mpnet-base-v2
  --save_dir group_results/ 
```
Then, you can apply batch prompting on the grouping results:
```bash
python run_batch_inference-group.py
  --exp_name batch_inference-commonsense_qa 
  --num_in_batch 2 
  --group_method similarity
  --openai_api_key openai_api_key.txt 
  --save_dir results/ 
  --use_cache True
```

## Code structure
Please refer to [HKUNLP/humanprompt](https://github.com/HKUNLP/HumanPrompt) README for code structure details.


## Citation
If you find this repo useful, please consider citing [HKUNLP/humanprompt](https://github.com/HKUNLP/HumanPrompt) project, [manifest](https://github.com/HazyResearch/manifest) project and our paper. Thank you!
```bibtex
@software{humanprompt,
  author = {Tianbao Xie and
            Zhoujun Cheng and
            Yiheng Xu and
            Peng Shi and
            Tao Yu},
  title = {A framework for human-readable prompt-based method with large language models},
  howpublished = {\url{https://github.com/hkunlp/humanprompt}},
  year = 2022,
  month = October
}
```

```bibtex
@misc{orr2022manifest,
  author = {Orr, Laurel},
  title = {Manifest},
  year = {2022},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/HazyResearch/manifest}},
}
```

```bibtex
@article{cheng2023batch,
  title={Batch Prompting: Efficient Inference with Large Language Model APIs},
  author={Cheng, Zhoujun and Kasai, Jungo and Yu, Tao},
  journal={arXiv preprint arXiv:2301.08721},
  year={2023}
}
```
