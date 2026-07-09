import random

from mvp.core.generation import generate_with_forced_steps
from mvp.core.model_setup import reward_completion
from mvp.core.records import StrategyRecord
from mvp.experiment.strategies import build_strategy_steps


def run_reference(model, prompt_ids, prompt_mask, sample, sample_idx, experiment, args):
    """生成一条 reference 轨迹并记录 delimiter candidates。"""

    reference_mode = experiment.reference_mode
    args.current_reference_mode = reference_mode
    baseline = generate_with_forced_steps(
        model,
        prompt_ids,
        prompt_mask,
        sample_idx=sample_idx,
        forced_steps=set(),
        args=args,
        prompt_augment=experiment.prompt_augment,
        collect_candidates=True,
    )
    reference_reward = reward_completion(baseline.completion, sample)
    candidates = baseline.candidates[: args.max_candidates_per_sample]
    for candidate in candidates:
        candidate.reference_reward = reference_reward
    return baseline, reference_reward, candidates


def run_strategy_rollouts(model, prompt_ids, prompt_mask, sample, sample_idx, reference_mode,
                          baseline, reference_reward, candidates, experiment, args):
    """探索性多点策略 rollout；主结论不要依赖这个指标。"""

    rng = random.Random(args.seed + sample_idx)
    strategy_to_steps = build_strategy_steps(candidates, args.budget, args.random_trials, rng)
    rows = [
        StrategyRecord(
            sample_idx=sample_idx,
            reference_mode=reference_mode,
            strategy="reference",
            reward=reference_reward,
            planned_steps=[],
            inserted_steps=[],
            completion=baseline.completion,
        )
    ]

    for strategy, steps in strategy_to_steps.items():
        trace = generate_with_forced_steps(
            model,
            prompt_ids,
            prompt_mask,
            sample_idx=sample_idx,
            forced_steps=set(steps),
            args=args,
            prompt_augment=experiment.prompt_augment,
            collect_candidates=False,
        )
        rows.append(
            StrategyRecord(
                sample_idx=sample_idx,
                reference_mode=reference_mode,
                strategy=strategy,
                reward=reward_completion(trace.completion, sample),
                planned_steps=steps,
                inserted_steps=trace.forced_steps_used,
                completion=trace.completion,
            )
        )
    return rows
