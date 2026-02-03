# InT: Self-Proposed Interventions Enable Credit Assignment in LLM Reasoning

To generate interventions:

1. Sample incorrect attempts and construct a dataset that has the following columns: `problem`, `answer`, `reference_solution`, `incorrect_attempt`
2. Give this dataset, ask the base model to (i) verify the solution and (ii) propose an intervention
    - Run `propose_interventions.sh` on the dataset pushed to hub
    - Parse the intervention `parse_interventions.ipynb`
    - This should give you the SFT data (see [example SFT set](https://huggingface.co/datasets/CMU-AIRe/InT-SFT))
4. SFT on (correct prefix + intervention | problem), i.e., with input as `problem` and output as `intervention_guided_attempt`
5. Online RL on the same set of problems (see [example RL set](https://huggingface.co/datasets/CMU-AIRe/InT-RL))

If you run into any problems with the code, please submit a Github issue!
