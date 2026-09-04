#!/usr/bin/env python3
"""Reproducible sweep for delimiter-only prompt hallucinations in Qwen3.5-9B."""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import (
    AutoModelForMultimodalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)


MODEL_ID = "Qwen/Qwen3.5-9B"

HIGH_YIELD_PROMPTS = ["___", "---", "##", "***", "<hr>", "1."]
ALL_PROMPTS = [
    "see the below\n---\n",
    "##",
    "----------",
    "---",
    "___",
    "===",
    "***",
    "- - -",
    "--",
    "-",
    "<hr>",
    "<hr/>",
    "...",
    "1.",
]

PRESETS = {
    "qwen_recommended": {
        "mode": "thinking_general",
        "enable_thinking": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
    "openrouter_conventional": {
        "mode": "thinking",
        "enable_thinking": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    },
}

INTERESTING_RE = re.compile(
    r"shirt|8\s*[x×]\s*8|grid|row(?:s)?\b|column(?:s)?\b|visual pattern|"
    r"equation|math(?:ematical)? problem|puzzle|code snippet|uploaded|screenshot|"
    r"multiple[ -]choice|sequence of numbers|image (?:you|the user)",
    re.IGNORECASE,
)


class PresencePenalty(LogitsProcessor):
    """Subtract a fixed penalty from every token already generated."""

    def __init__(self, penalty: float, prompt_length: int):
        self.penalty = penalty
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.prompt_length :]
        if self.penalty and generated.shape[1]:
            for batch_index in range(generated.shape[0]):
                seen = generated[batch_index].unique()
                scores[batch_index, seen] -= self.penalty
        return scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-prompt", type=int, default=6)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--full", action="store_true", help="Use all delimiter prompts")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="openrouter_conventional",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    config = PRESETS[args.preset]
    prompts = ALL_PROMPTS if args.full else HIGH_YIELD_PROMPTS
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path(
        f"/workspace/mats12/results/qwen_{args.preset}_{timestamp}.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID} in BF16...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    print(
        f"Loaded on {model.device}; preset={args.preset}; running {len(prompts)} prompts x "
        f"{args.samples_per_prompt} seeds.",
        flush=True,
    )

    interesting_count = 0
    record_count = 0
    with output.open("w", encoding="utf-8") as handle:
        for prompt_index, prompt in enumerate(prompts):
            messages = [{"role": "user", "content": prompt}]
            serialized = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=config["enable_thinking"],
            )
            inputs = tokenizer(
                serialized,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(model.device)
            prompt_length = inputs["input_ids"].shape[-1]

            for sample_index in range(args.samples_per_prompt):
                seed = args.seed_start + prompt_index * args.samples_per_prompt + sample_index
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=config["temperature"],
                        top_p=config["top_p"],
                        top_k=config["top_k"],
                        min_p=config["min_p"],
                        repetition_penalty=config["repetition_penalty"],
                        logits_processor=LogitsProcessorList(
                            [PresencePenalty(config["presence_penalty"], prompt_length)]
                        ),
                        use_cache=True,
                    )

                completion_ids = generated_ids[0, prompt_length:]
                completion = tokenizer.decode(
                    completion_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                interesting = bool(INTERESTING_RE.search(completion))
                interesting_count += int(interesting)
                record_count += 1

                record = {
                    "model_id": MODEL_ID,
                    "prompt": prompt,
                    "serialized_prompt": serialized,
                    "seed": seed,
                    "config": config,
                    "max_new_tokens": args.max_new_tokens,
                    "generated_token_count": int(completion_ids.shape[0]),
                    "interesting_heuristic": interesting,
                    "completion": completion,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

                marker = "INTERESTING" if interesting else "ordinary"
                preview = completion.replace("\n", " ")[:180]
                print(
                    f"[{record_count:02d}] prompt={prompt!r} seed={seed} "
                    f"tokens={completion_ids.shape[0]} {marker}: {preview}",
                    flush=True,
                )

    print(
        f"Done: {record_count} samples, {interesting_count} heuristic hits.\n"
        f"Raw results: {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
