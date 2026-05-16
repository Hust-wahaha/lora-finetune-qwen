# Project Progress Log

## 2026-05-15

- Established reproducible workspace layout: `data/raw`, `data/interim`, `data/final`, `scripts`, `runs`, `artifacts`.
- Added pilot dataset generator and validator.
- Produced `data/final/pilot.jsonl` and validated schema successfully.
- Confirmed `ms-swift` supports local `jsonl/json/csv` datasets.
- Reworked the training entrypoint to consume local `messages`-style JSONL data.
- Expanded to a structured-coT-centered v1 dataset and completed a first formal training run.
- Generated the current standard `s800` dataset with 400 aligned source items and 800 view-expanded train/val/test records.
- Completed the main `s800` LoRA run at `runs/20260515_210429/checkpoints/checkpoint-40`.

## Agreed Data Direction

- Keep `structured_cot` as the current main supervision target because it is the most controllable, easiest to validate, and most aligned with the near-term experiment goal.
- Treat the current standard training task as two input views mapping to the same target style:
  - `modern_question -> structured_cot + answer`
  - `classical_question -> structured_cot + answer`
- Do **not** treat the base model's natural `<think>` as `modern_cot` training labels.
- Reserve `modern_cot` / `classical_cot` as future supervised targets for later ablation, separate from baseline natural reasoning traces.
- Use the current large-scale test to answer one narrow question first: whether structured supervision plus classical/modern input views improves behavior relative to baseline.

## 2026-05-16 Evaluation Update

- Ran a first full comparison on `data/final/test_s800.jsonl` with `max_tokens=256`.
- Observed `baseline=23.75%` and `finetuned=63.75%`, but this run was **not reliable enough for final judgment** because long `<think>` traces often consumed the decode budget before the final answer was emitted.
- Confirmed the truncation issue by inspecting `weight_remaining` predictions: many responses contained correct intermediate arithmetic but were cut before a stable final answer string, which corrupted answer extraction.
- Re-ran the same comparison with `max_tokens=512` and a slightly more robust extraction rule at `runs/20260516_150218_compare_s800_full_mt512`.

### Corrected Rule-Based Full-Test Result

- baseline: `44/80 = 55.0%`
- finetuned: `77/80 = 96.25%`

### Corrected Rule-Based Finetuned Per-Family Accuracy

- `give_away`: `100%`
- `box_total`: `100%`
- `equal_division`: `85.7%`
- `unit_price`: `92.9%`
- `bundle_count`: `100%`
- `weight_remaining`: `100%`

### Corrected Rule-Based Finetuned View Accuracy

- modern: `100%`
- classical: `92.5%`

## Current Interpretation

- The current `s800` structured-coT dataset is directionally very strong on its own synthetic distribution.
- The main residual problem is no longer basic arithmetic correctness; it is output control.
- Both baseline and finetuned models still retain the Qwen default `<think>` behavior, which causes:
  - verbose generations
  - unstable final-answer emission
  - evaluation sensitivity to decode length
- Therefore, the next iteration must optimize both:
  - data diversity
  - target-format discipline
  - evaluation robustness

## 2026-05-16 Evaluation Pipeline Update

- The formal evaluation pipeline is now upgraded to a two-stage design:
  - Stage 1: rule-based numeric answer extraction and exact-match scoring
  - Stage 2: `DeepSeek V4 Flash` review for mismatched or selected samples
- This change is motivated by a concrete failure mode already observed in baseline outputs:
  - the model may reason correctly
  - the final answer may be buried inside a long response
  - rule-based extraction may therefore underestimate true answer correctness
- The new standard evaluation script is `scripts/eval_compare_full.py`.
- It now supports:
  - direct generation + evaluation
  - reusing existing prediction files from an older run
  - `DeepSeek V4 Flash` review for `mismatches` or `all` samples
  - separate reporting of rule accuracy and LLM-reviewed final accuracy

## 2026-05-16 Full DeepSeek Review Result

- Reused the saved `512`-token prediction files and ran full mismatch review at `runs/20260516_full_llm_review`.
- Review setting:
  - rule stage first
  - `DeepSeek V4 Flash` reviews all rule-mismatch cases
- Baseline review outcome:
  - rule-based: `44/80 = 55.0%`
  - LLM-reviewed final: `72/80 = 90.0%`
  - reviewed mismatch cases: `36`
  - changed from incorrect to correct: `28`
- Finetuned review outcome:
  - rule-based: `77/80 = 96.25%`
  - LLM-reviewed final: `80/80 = 100%`
  - reviewed mismatch cases: `3`
  - changed from incorrect to correct: `3`

### LLM-Reviewed Baseline Final Accuracy

- by family:
  - `give_away`: `78.6%`
  - `box_total`: `100%`
  - `equal_division`: `100%`
  - `unit_price`: `64.3%`
  - `bundle_count`: `100%`
  - `weight_remaining`: `100%`
- by view:
  - modern: `100%`
  - classical: `80%`

### LLM-Reviewed Finetuned Final Accuracy

- by family: all `100%`
- by view:
  - modern: `100%`
  - classical: `100%`

## Updated Conclusion

- The gap between rule-based scoring and LLM-reviewed scoring is very large for the baseline model, which proves that baseline absolute accuracy was heavily underestimated by numeric extraction alone.
- The finetuned model is much less affected by this issue, because it is more likely to emit an explicit and stable final answer.
- Therefore, later reports should present **both**:
  - rule-based accuracy
  - LLM-reviewed final accuracy
- For the current `s800` task distribution, the mainline LoRA model is now best described as:
  - rule-based strong
  - LLM-reviewed essentially saturated
  - still weak in output control rather than problem solving

## Immediate Next Steps

- Keep `s800` as the current in-distribution baseline dataset.
- Use `scripts/eval_compare_full.py` as the standard full comparison entrypoint.
- Report at least three metrics in all later experiments:
  - `Rule-based Accuracy`
  - `LLM-reviewed Final Accuracy`
  - `Answer Marker Rate`
- Use `DeepSeek V4 Flash` as the standard lightweight review model for difficult evaluation cases.
- Prepare the next dataset iteration to improve classical robustness and reduce reliance on long post-training `<think>` traces.
