import argparse, pickle, os
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams


def apply_template(problem):
    messages = [
        {"role": "user", "content": problem}
    ]
    return messages

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset_name", type=str)
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--dataset_split", type=str)
    parser.add_argument("--dataset_start", type=int)
    parser.add_argument("--dataset_end", type=int)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("-K", type=int)
    parser.add_argument("--model", type=str)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)

    args = parser.parse_args()
    print(args.dataset_start, args.dataset_end)
    print(args.model)
    print(args.output_path)
    print("Top K:", args.top_k)

    os.makedirs(args.output_path, exist_ok=True)


    dataset = load_dataset(args.input_dataset_name, split=args.dataset_split,  data_dir=args.subset)
    
    print(f"Filtered dataset to {len(dataset)} examples.")

    args.dataset_end = min(args.dataset_end, len(dataset))
    if args.dataset_start >= len(dataset):
        raise ValueError(f"dataset_start {args.dataset_start} must be less than the length of the dataset {len(dataset)}")
    elif args.dataset_start >= args.dataset_end:
        raise ValueError(f"dataset_start {args.dataset_start} must be less than dataset_end {args.dataset_end}")
    
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, enable_prefix_caching=True, max_model_len=32768)

    solutions = {}
    for i in tqdm(range(args.dataset_start, args.dataset_end, args.batch_size)):
        batch_problems = dataset[i : min(i + args.batch_size, args.dataset_end)]['problem']
        convs = [apply_template(problem) for problem in batch_problems]
        
        completions = llm.chat(
            messages=convs,
            sampling_params=SamplingParams(
                n=args.K,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=None,
            ),
        )
        for j, completion in enumerate(completions):
            solutions[i + j] = completion

    with open(os.path.join(args.output_path, f'pass_at_{args.K}_{args.dataset_start}_{args.dataset_end}_{args.dataset_split}.pkl'), 'wb') as f:
        pickle.dump(solutions, f)
