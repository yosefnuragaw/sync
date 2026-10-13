from typing import Dict, List

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.prompts import PromptProcessor


def init_judge(model_name: str) -> tuple[LLM, AutoTokenizer]:
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        trust_remote_code=True,
        # The judge prompt alone runs ~1300-1600 tokens, and a reasoning judge needs
        # room for its analysis channel and the <reasoning>/<score> block after it.
        max_model_len=8192,
        gpu_memory_utilization=0.9,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return llm, tokenizer


def read_answers(behavior: str, path: str) -> List[Dict[str, str]]:
    dataset = load_dataset("csv", data_files=path, split='train')
    prompts = []
    if behavior == 'jailbreak':
        for row in dataset:
            prompts.append({
                'behavior': behavior,
                'question': row.get('questions', row.get('question', '')),
                'answer': row.get('answers', row.get('answer', '')),
                'positive_example': None,
                'negative_example': None
            })
    else:
        for row in dataset:
            if row.get('matching') == 'A':
                pos, neg = row.get('A', ''), row.get('B', '')
            else:
                pos, neg = row.get('B', ''), row.get('A', '')
            prompts.append({
                'behavior': behavior,
                'question': row.get('questions', row.get('question', '')),
                'answer': row.get('answers', row.get('answer', '')),
                'positive_example': pos,
                'negative_example': neg
            })
    return prompts


def evaluate_batch(
    llm: LLM,
    tokenizer: AutoTokenizer,
    factory: PromptProcessor,
    prompts: List[str],
    desc: str,
) -> float:
    # Greedy decoding, so the judge returns the same score for the same text every run.
    # max_tokens has to cover the analysis channel as well as the verdict: too small a
    # budget runs out mid-analysis and the reply parses as -1 with no <score> tag.
    sampling_params = SamplingParams(
        max_tokens=4096,
        min_tokens=1,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )

    chat_prompts = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        chat_prompts.append(formatted_prompt)

    outputs = llm.generate(chat_prompts, sampling_params)

    start = len(factory.score)
    truncated = 0
    for out in tqdm(outputs, desc=f"Parsing {desc} Results"):
        if out.outputs[0].finish_reason == "length":
            truncated += 1
        factory.read_response(out.outputs[0].text)

    # Report the loss rather than averaging the survivors silently: a cell that lost
    # most of its rows would otherwise look like a cell that scored fine.
    scored = factory.score[start:]
    unparsed = sum(1 for s in scored if s == -1.0)
    if unparsed or truncated:
        print(f"  [{desc}] {unparsed}/{len(scored)} unparsed ({unparsed/len(scored):.0%}), "
              f"{truncated} hit the token limit", flush=True)
        if unparsed / max(len(scored), 1) > 0.05:
            print(f"  [{desc}] WARNING: over 5% unparsed. A per-prompt slope needs EVERY "
                  f"multiplier scored, so this compounds across the sweep.", flush=True)

    valid_scores = [s for s in scored if s != -1.0]
    if not valid_scores:
        return 0.0
    return sum(valid_scores) / len(valid_scores)


def get_model_dir(model_name_or_path: str) -> str:
    if 'gemma' in model_name_or_path:
        return 'gemma'
    elif 'llama' in model_name_or_path:
        return 'llama'
    else:
        return 'mistral'
