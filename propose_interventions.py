import argparse, pickle, os
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

SOLVE_PROMPT = lambda problem, reference_solution: """You are an expert mathematician teaching a Math Olympiad class. You will be given a problem and a high-level reference solution to the problem. Your task is to solve the problem step-by-step guided by the high-level reference solution.

# Problem #
""" + problem + """

# High-Level Oracle Solution #
""" + reference_solution

SELF_CORRECT_PROMPT = lambda incorrect_attempt, correct_answer: f"""Great job! Now, a student in your class has solved the problem incorrectly. You must leverage your understanding of the reference solution to help him correct his attempt at the problem.

### Detailed Instructions ###

**1. Detailed Verification Log**

You must perform a **step-by-step** check of the student's attempt against the oracle solution. The steps here refer to each numbered substep in the **Incorrect Student Attempt** (i.e., **Substep 1:** ..., **Substep 2:** ..., ...), and not the high-level steps (e.g., ### Step 1: ...) which may contain multiple numbered substeps. This analysis will be presented in a **Detailed Verification Log**, where you justify your assessment of each substep in bullet points: for correct substeps, a brief justification suffices; for substeps with errors or gaps, you must provide a detailed explanation. **Be careful and check every intermediate result, they are very easy to miss.**

**2. Identify the First Critical Error**

For each issue in the detailed verification log, you MUST determine whether it is a **critical error**. A critical error must pass the following two checks:

1. A critical error is either a **factual error** (e.g., a calculation error like `2+3=6`) or **logical fallacy** (e.g., claiming that `A>B, C>D` implies `A-C>B-D`) that disrupts the current line of reasoning. 
    *   **Procedure:** To perform the first check, explain the specific error and state that it **invalidates the current line of reasoning**.
2. A critical error must not be recovered from. 
    *   **Procedure:** You must double-check that the error is indeed not recovered from in later steps, i.e., there does not exist a later statement that says something like "Wait, but let me double-check this claim..." and goes on to dispute the error.

As long as the issue passes the two checks above, it is considered a **critical error**. We are interested in the *first* critical error that the student makes.

**3. Propose An Intervention Step**

After finding the critical error, you should propose an **intervention step** that will be inserted before the occurrence of the critical error to steer the student towards the correct solution. You should frame the intervention step in a way that **does not give away the exact answer**, so that the student can still learn from the error. More importantly, you should provide detailed guidance on exactly what to try next by **showing a sketch of the next steps that will lead him to the correct solution**.

**4. Output Format**

Your response MUST be structured into three main sections: a **Detailed Verification Log**, followed by a **Critical Error Report**, and finally an **Intervention Step**.

**4.1 Detailed Verification Log**

Provide the full, step-by-step verification log as defined in the Detailed Instructions, structured in bullet points. When you refer to a specific part of the student's attempt or the oracle solution, **quote the relevant text** to make your reference clear before providing your detailed analysis of that part.

**4.2 Critical Error Report**

In this report, you should first include a bulleted list that summarizes **every** issue you discovered. For each issue, you must provide:
    
1. **Location:** A direct quote of the key phrase or equation where the issue occurs.
2. **Issue:** A brief description of the problem and whether or not is a **Critical Error** that passes the two checks listed in **Detailed Instructions**.

You should stop once you have found the **first** critical error.

**4.3 Intervention Step**

Finally, you should provide a single intervention step to be inserted before the occurrence of the critical error that will steer the student towards the correct solution. You should specify the location in the student's attempt to insert the intervention step. **Write the content of the intervention step from the student's perspective. The student should be able to continue from the inserted intervention step without feeling that someone else wrote it.** If you believe the student's solution is on the right track and there are no critical errors, leave the intervention step empty.

Follow the format below to write a draft of the intervention step first. Perform a self-check after writing the draft, and if you are satisfied with the draft, write the final intervention step.

**Draft of the intervention step format:**

1. **Content:** Content of the intervention step.
2. **Location:** The substep number in the student's attempt to insert the intervention step. You must **quote the relevant text** from the student's attempt to support your choice of location.

**Self-check format:**

Answer each of the three questions below with two parts: (1) a brief explanation of your reasoning, and (2) a single final verdict — Yes or No.

1. Does the **content** of the intervention step include the exact answer from the oracle solution (i.e., "{correct_answer}") anywhere?
2. Is the **content** of the intervention step missing any key insights that are necessary to solve the problem?
3. Is the **location** of the intervention step later than the substep number that contains the first critical error?

If the answer is "Yes" to **any** of the questions, the intervention step draft has **failed the self-check**. Revise the draft so that it passes the self-check, and perform a self-check again. Repeat this process until the draft passes the self-check.

If the answer is "No" to **all** of the questions, the intervention step draft has **passed the self-check**. In this case, write the final intervention step in the format below.

**Final intervention step format:**

INSERT_STEP_CONTENT should be the content of the intervention step to be inserted, INSERT_STEP_NUMBER should be an integer without any extra text indicating the substep number in the student's attempt to insert the intervention step)

<intervention>
{{ "content": INSERT_STEP_CONTENT, "location": INSERT_STEP_NUMBER }}
</intervention>

# Incorrect Student Attempt #
""" + incorrect_attempt


def apply_solve_template(problem, oracle_solution):
    messages = [
        {"role": "user", "content": SOLVE_PROMPT(problem, oracle_solution)}
    ]
    return messages

def apply_self_correct_template(problem, oracle_solution, correct_solution, student_solution, correct_answer):
    lines = student_solution.split("\n\n")
    numbered_lines = [f"**Substep {i + 1}:** {line}" for i, line in enumerate(lines)]
    numbered_student_solution = "\n\n".join(numbered_lines)
    messages = [
        {"role": "user", "content": SOLVE_PROMPT(problem, oracle_solution)},
        {"role": "assistant", "content": correct_solution},
        {"role": "user", "content": SELF_CORRECT_PROMPT(numbered_student_solution, correct_answer)}
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
    parser.add_argument("--top_k", type=int, default=20)
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
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    solutions = {}
    for i in tqdm(range(args.dataset_start, args.dataset_end, args.batch_size)):
        batch_problems = dataset[i : min(i + args.batch_size, args.dataset_end)]['problem']
        batch_reference_solutions = dataset[i : min(i + args.batch_size, args.dataset_end)]['reference_solution']
        batch_incorrect_attempts = dataset[i : min(i + args.batch_size, args.dataset_end)]['incorrect_attempt']
        batch_correct_answers = dataset[i : min(i + args.batch_size, args.dataset_end)]['answer']
       
        # ask the model to solve the problem first
        convs = [
            tokenizer.apply_chat_template(apply_solve_template(problem, reference_solution), tokenize=False, add_generation_prompt=True)
            for problem, reference_solution in zip(batch_problems, batch_reference_solutions)
        ]
        
        completions = llm.generate(
            prompts=convs,
            sampling_params=SamplingParams(
                n=1,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                top_k=args.top_k,
            ),
        )

        # ask the model to self-correct the student's solution
        convs = []
        for problem, reference_solution, completion, incorrect_attempt, correct_answer in zip(batch_problems, batch_reference_solutions, completions, batch_incorrect_attempts, batch_correct_answers):
            conv = tokenizer.apply_chat_template(apply_self_correct_template(problem, reference_solution, completion.outputs[0].text, incorrect_attempt, correct_answer), tokenize=True, add_generation_prompt=True)
            convs.append(tokenizer.decode(conv[:30000]))

        completions = llm.generate(
            prompts=convs,
            sampling_params=SamplingParams(
                n=args.K,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                top_k=args.top_k,
            ),
        )
        
        for j, completion in enumerate(completions):
            solutions[i + j] = completion

    with open(os.path.join(args.output_path, f'interventions_{args.dataset_start}_{args.dataset_end}.pkl'), 'wb') as f:
        pickle.dump(solutions, f)