from dataclasses import dataclass, field
from typing import Dict, List, Optional
import argparse
from transformers import HfArgumentParser
import os
from collections import defaultdict
from utils.prompts import PromptProcessor
from utils.common import set_seed
from utils.judge import init_judge, read_answers, evaluate_batch, get_model_dir

PERSONA_BEHAVIOR_MAP = {
    'persona-1': ['power-seeking', 'wealth-seeking', 'corrigible-less', 'coordinate-other'],
    'single-ablation':['power-seeking','wealth-seeking'],
    'persona-2': ["machiavellianism", "narcissism","psychopathy"],
    'persona-3': ["evil", "hallucinating", "sycophantic"],
    # persona-3 pair subsets; each is scored on its own 2x2 cross-rubric block.
    'persona-3-ab': ["evil", "hallucinating"],
    'persona-3-ac': ["evil", "sycophantic"],
    'persona-3-bc': ["hallucinating", "sycophantic"],
    # Attribute-subset sweeps. Which attributes are STEERED lives in each config's
    # vec_dir, not here: every run is scored on the full behaviour list.
    'cross':   ['power-seeking', 'wealth-seeking', 'corrigible-less', 'coordinate-other'],
    'cross-3': ["evil", "hallucinating", "sycophantic"],
}


@dataclass
class ScriptArguments:
    id: Optional[str] = field(default="baseline", metadata={"help": "Run id"})
    model_name_or_path: Optional[str] = field(default="google/gemma-3-1b-it", metadata={"help": "Model Answer Folder"})
    judge_name: Optional[str] = field(default="openai/gpt-oss-20b", metadata={"help": "Judge Model id"})
    behavior: Optional[str] = field(default="power-seeking", metadata={"help": "the behavior"})
    vec_dir: Optional[List[str]] = field(default=None, metadata={"help": "Read only to resolve steered_only; scoring itself does not steer"})
    steered_only: Optional[bool] = field(default=False, metadata={"help": "Score ONLY the behaviours named in vec_dir. Must match the flag the answers were generated under, or the missing prompt sets will not be on disk"})
    multipliers: Optional[List[float]] = field(default_factory=lambda: [2., 1.5, 1., 0.5, -0.5, -1., -1.5, -2.], metadata={"help": "the multipliers evaluated"})
    answer_dir: Optional[str] = field(default="generation_results/gemma3-1b", metadata={"help": "Directory where answers are saved"})


def main(baseline: bool, args: ScriptArguments, utility_only: bool = False,
         rubrics: Optional[List[str]] = None, skip_utility: bool = False) -> None:
    """`rubrics` narrows the rubric axis only, leaving the prompt axis whole.

    Every cell is written to its own reasoning_<prompt>_<rubric>_behavior_mul_<m>.csv, so
    naming a rubric rewrites exactly the files that rubric owns. The prompt axis stays
    whole because the off-diagonal cells are what cross_steerability is computed from.
    """
    llm, tokenizer = init_judge(args.judge_name)

    # Key: "prompt_behav_vs_rubric_behav" -> {multiplier -> score}
    accuracy_likert: Dict[str, Dict[float, float]] = defaultdict(lambda: defaultdict(float))
    coherence_likert: Dict[float, float] = {}
    model_dir = get_model_dir(args.model_name_or_path)

    if args.behavior in PERSONA_BEHAVIOR_MAP.keys():
        behaviors = PERSONA_BEHAVIOR_MAP[args.behavior]
    else:
        behaviors = [args.behavior]

    # steered_only: the generation run wrote prompt sets only for the behaviours in
    # vec_dir, so both axes shrink to that list, leaving an NxN block over the steered
    # subset. `behavior` is unchanged, so reasoning_base is the same directory.
    if args.steered_only and not baseline:
        steered = {os.path.normpath(v).split(os.sep)[-3] for v in (args.vec_dir or [])}
        kept = [b for b in behaviors if b in steered]
        if not kept:
            raise ValueError(
                f"steered_only left nothing to score: behavior '{args.behavior}' expands "
                f"to {behaviors}, vec_dir steers {sorted(steered)}."
            )
        print(f"[steered_only] scoring {kept}; skipping {[b for b in behaviors if b not in steered]}")
        behaviors = kept

    if "generation_results" in args.answer_dir:
        base_dir = args.answer_dir.split("generation_results")[0]
    else:
        base_dir = os.path.dirname(args.answer_dir.rstrip("/"))

    reasoning_base = os.path.join(base_dir, "reasoning", args.behavior, model_dir)

    # --- Load datasets ---
    if baseline:
        # Load each behavior's own baseline.csv (one per behavior subfolder)
        datasets = defaultdict(lambda: defaultdict(list))
        for behav in behaviors:
            file_path = f"{args.answer_dir}/{behav}/baseline.csv"
            datasets[behav][0] = read_answers(behavior=behav, path=file_path)
    else:
        datasets = defaultdict(lambda: defaultdict(list))
        for behav in behaviors:
            for multiplier in args.multipliers:
                file_path = f"{args.answer_dir}/{behav}/{multiplier}.csv"
                datasets[behav][multiplier] = read_answers(behavior=behav, path=file_path)

    # --- Utility: evaluated once per multiplier, across ALL behaviors' datasets combined ---
    datasets_by_mul: Dict[float, List[Dict]] = defaultdict(list)
    for behav, behav_dataset in datasets.items():
        for mul, dataset in behav_dataset.items():
            datasets_by_mul[mul].extend(dataset)

    for mul in ([] if skip_utility else ([0] if baseline else args.multipliers)):
        dataset = datasets_by_mul[mul]
        print(f"\n[Evaluating Utility | Multiplier: {mul}]")
        sub_dir = 'baseline' if mul == 0 else args.id  # match the behavior loop below

        coherence_factory = PromptProcessor()
        coherence_raw_prompts = []
        for row in dataset:
            coherence_raw_prompts.append(coherence_factory.produce_prompt(
                behavior="utility",
                question=row['question'],
                answer=row['answer'],
                positive_example=None,
                negative_example=None
            ))

        c_score = evaluate_batch(llm, tokenizer, coherence_factory, coherence_raw_prompts, "Utility")
        coherence_likert[mul] = c_score
        coherence_factory.save(os.path.join(reasoning_base, sub_dir, f"reasoning_utility_mul_{mul}.csv"))
        print(f'Utility Score (Scale 4): {coherence_likert[mul]:.2f}')

    if utility_only:
        print(f'\nFinal Utility Likert:  {coherence_likert}')
        return

    # Narrow the rubric axis after loading: rubric_examples below still needs
    # datasets[rubric_behav], and an unscored rubric is still a prompt set.
    rubric_behaviors = behaviors
    if rubrics:
        unknown = [r for r in rubrics if r not in behaviors]
        if unknown:
            raise ValueError(f"--rubrics {unknown} not in {behaviors} for behavior '{args.behavior}'")
        rubric_behaviors = [b for b in behaviors if b in rubrics]
        print(f"[rubrics] scoring rubric axis {rubric_behaviors}; "
              f"leaving {[b for b in behaviors if b not in rubric_behaviors]} on disk untouched")

    # --- Behavior: each prompt-behavior dataset is evaluated against ALL behavior rubrics ---
    for prompt_behav, behav_dataset in datasets.items():
        for mul, dataset in behav_dataset.items():
            sub_dir = 'baseline' if mul == 0 else args.id

            for rubric_behav in rubric_behaviors:
                print(f"\n[Evaluating Prompt Behavior: {prompt_behav} | Rubric: {rubric_behav} | Multiplier: {mul}]")

                # Use positive/negative examples from the rubric behavior's dataset at the same multiplier
                rubric_dataset = datasets[rubric_behav][mul]
                rubric_examples = {row['question']: row for row in rubric_dataset}

                behavior_factory = PromptProcessor()
                behavior_raw_prompts = []
                for row in dataset:
                    rubric_row = rubric_examples.get(row['question'], row)
                    behavior_raw_prompts.append(behavior_factory.produce_prompt(
                        behavior=rubric_behav,
                        question=row['question'],
                        answer=row['answer'],
                        positive_example=rubric_row['positive_example'],
                        negative_example=rubric_row['negative_example']
                    ))

                b_score = evaluate_batch(llm, tokenizer, behavior_factory, behavior_raw_prompts, "Behavior")
                accuracy_likert[f"{prompt_behav}_vs_{rubric_behav}"][mul] = b_score
                behavior_factory.save(os.path.join(
                    reasoning_base, sub_dir,
                    f"reasoning_{prompt_behav}_{rubric_behav}_behavior_mul_{mul}.csv"
                ))
                print(f'Behavior Score [{prompt_behav} x {rubric_behav}] (Scale 4): {b_score:.2f}')

    print(f'\nFinal Accuracy Likert: {dict(accuracy_likert)}')
    print(f'Final Utility Likert:  {coherence_likert}')


if __name__ == "__main__":
    set_seed(seed=42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to your YAML config file")
    parser.add_argument("--baseline", action='store_true', help="Run only the baseline (multiplier 0)")
    parser.add_argument("--utility-only", action='store_true',
                        help="Score utility only, leaving the existing behavior CSVs untouched")
    parser.add_argument("--rubrics", nargs="+", default=None,
                        help="Score ONLY these rubric behaviours, e.g. --rubrics sycophantic "
                             "hallucinating. The prompt axis is unchanged, so every cell that "
                             "uses the named rubric is rewritten and no other file is touched.")
    parser.add_argument("--skip-utility", action='store_true',
                        help="Skip the utility pass. Use with --rubrics for a rescoring run: "
                             "utility does not depend on the behaviour rubrics.")
    args, _ = parser.parse_known_args()

    hf_parser = HfArgumentParser(ScriptArguments)
    if args.config.endswith(".yaml"):
        script_args = hf_parser.parse_yaml_file(yaml_file=args.config, allow_extra_keys=True)[0]
    elif args.config.endswith(".json"):
        script_args = hf_parser.parse_json_file(json_file=args.config, allow_extra_keys=True)[0]
    else:
        raise ValueError("Config file must be .yaml or .json")

    main(baseline=args.baseline, args=script_args, utility_only=args.utility_only,
         rubrics=args.rubrics, skip_utility=args.skip_utility)