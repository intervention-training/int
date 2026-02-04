# InT: Self-Proposed Interventions Enable Credit Assignment in LLM Reasoning

We provide instructions to reproduce our results below. As you run the `.sh` and `.ipynb` files, fill in your local paths where appropriate. 

We run our experiments on [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).

### Step 0: Curate data and setup environment
We provide a [dataset](https://huggingface.co/datasets/CMU-AIRe/InT-hard-set) of **very** difficult problems sourced from various datasets, mostly selected by filtering for those with zero reward under 64 rollouts.

Install vllm, datasets, transformers, math-verify, and tqdm in a Python 3.10 experiment.

### Step 1: Sample incorrect rollouts
Run `sample_rollouts.sh` then `filter_rollouts.ipynb`.

It'll sample rollouts, filter for those that are incorrect, and construct a dataset that has the following columns: `problem`, `answer`, `reference_solution`, `incorrect_attempt`.

The end result should look like [this](https://huggingface.co/datasets/CMU-AIRe/InT-hard-set-with-incorrect-attempts).

### Step 2: Propose interventions
Run `propose_interventions.sh` then `parse_interventions.ipynb`.

It'll ask the base model to (i) verify the incorrect attempts and (ii) propose an intervention, and finally parse the interventions.

The end result should look like [this](https://huggingface.co/datasets/CMU-AIRe/InT-hard-set-with-incorrect-attempts-and-interventions).

### Step 3: Filter interventions (Optional)
Run `sample_guided_rollouts.sh` then `filter_guided_rollouts.ipynb`.

It'll select for interventions that actually lead to correct outcomes, which will serve as the SFT data.

The end result should look like [this](https://huggingface.co/datasets/CMU-AIRe/InT-SFT).

### Step 4: SFT on interventions
Perform SFT with [Llama Factory](https://github.com/hiyouga/LlamaFactory). Add the following in `dataset_info.json` and run `sft.sh`:

```
"int_train": {
    "hf_hub_url": "https://huggingface.co/datasets/CMU-AIRe/InT-SFT",
    "split": "train",
    "columns": {
        "prompt": "problem",
        "response": "intervention_guided_attempt"
    }
},
```

It'll perform SFT on (correct prefix + intervention | problem).

### Step 5: RL
Run online RL on the same [deduplicated set of problems](https://huggingface.co/datasets/CMU-AIRe/InT-RL). We use [Pipline RL](https://github.com/ars22/pipeline-rl) with `int.yaml` as the config. Add to `pipelinerl/domains/math/load_datasets.py` the following:


```
def process_int(dataset, dataset_name):
    for item in dataset:
        yield {
            "dataset": "item['source']",
            "task": item['problem'],
            "answer": "\\boxed{" + item['answer'] + "}",
        }
```

and in the `load_datasets` function, add

```
if "int" in dataset_names:
    dataset = load_dataset("CMU-AIRe/InT-RL", split="train", trust_remote_code=True)
    samples = [s for s in process_int(dataset, "int") if s is not None]
    logger.info(f"Loading Int dataset: {len(samples)} samples")
    datasets += add_ids(samples)
```

### Step 6: Eval
Run `eval.sh`.

It'll evaluate the model on [our hard eval dataset](https://huggingface.co/datasets/CMU-AIRe/InT-test-set).

---

If you run into any problems with reproducing the code, please submit a Github issue!
