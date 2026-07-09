from mvp.core.generation import generate_with_forced_steps
from mvp.core.model_setup import reward_completion
from mvp.core.records import GroupResultRecord, SelectedPointRecord
from mvp.experiment.point_selectors import score_of, select_points_for_group


def evaluate_counterfactual_groups(model, prompt_ids, prompt_mask, sample, sample_idx: int,
                                   reference_mode: str, reference_reward: float,
                                   candidates, sequence_points, experiment, args):
    """评估配置中的所有 candidate/sequence 单点反事实组。

    每个被选中的 step 都单独跑一次“只在这个 step 插入 latent”的 branch。
    这样得到的是点级别 utility，而不是多点策略 rollout reward。
    """

    selected_rows: list[SelectedPointRecord] = []
    group_rows: list[GroupResultRecord] = []
    branch_cache: dict[tuple[int, bool], tuple[float, list[int], bool]] = {}

    for group in experiment.groups:
        points = select_points_for_group(group, candidates, sequence_points)
        group_selected = []
        for rank, point in enumerate(points):
            row = make_selected_row(point, group, rank, reference_mode, reference_reward)
            fill_branch_result(
                row, branch_cache, model, prompt_ids, prompt_mask, sample,
                sample_idx, experiment.prompt_augment, args
            )
            selected_rows.append(row)
            group_selected.append(row)
        group_rows.append(make_group_result(sample_idx, reference_mode, group.name, group_selected))
    return selected_rows, group_rows


def make_selected_row(point, group, rank: int, reference_mode: str,
                      reference_reward: float) -> SelectedPointRecord:
    """把一个原始 candidate/sequence point 包装成待评估记录。"""

    return SelectedPointRecord(
        sample_idx=point.sample_idx,
        reference_mode=reference_mode,
        group=group.name,
        source=group.source,
        step=point.step,
        rank_in_group=rank,
        score_name=group.score,
        score=score_of(point, group.score),
        requires_delimiter=bool(group.requires_delimiter),
        reference_reward=reference_reward,
    )


def fill_branch_result(row: SelectedPointRecord, branch_cache, model, prompt_ids,
                       prompt_mask, sample, sample_idx: int, prompt_augment: bool, args) -> None:
    """运行或复用单点插入 branch，并把 reward/utility 写回 row。"""

    cache_key = (row.step, row.requires_delimiter)
    if cache_key not in branch_cache:
        branch = generate_with_forced_steps(
            model,
            prompt_ids,
            prompt_mask,
            sample_idx=sample_idx,
            forced_steps={row.step},
            forced_step_requires_delimiter={row.step: row.requires_delimiter},
            args=args,
            prompt_augment=prompt_augment,
            collect_candidates=False,
        )
        reward = reward_completion(branch.completion, sample)
        inserted = row.step in branch.forced_steps_used
        branch_cache[cache_key] = (reward, branch.forced_steps_used, inserted)

    reward, forced_steps_used, inserted = branch_cache[cache_key]
    row.branch_reward = reward
    row.utility = reward - row.reference_reward
    row.branch_forced_steps_used = forced_steps_used
    row.branch_inserted = inserted


def make_group_result(sample_idx: int, reference_mode: str, group_name: str,
                      rows: list[SelectedPointRecord]) -> GroupResultRecord:
    """一个 sample 内的组级结果；没选中点时 utility 按 0 计。"""

    if not rows:
        return GroupResultRecord(sample_idx, reference_mode, group_name, 0, 0, 0.0, 0.0)
    utilities = [float(row.utility or 0.0) for row in rows]
    inserted_count = sum(bool(row.branch_inserted) for row in rows)
    positives = sum(value > 0.0 for value in utilities)
    return GroupResultRecord(
        sample_idx=sample_idx,
        reference_mode=reference_mode,
        group=group_name,
        selected_count=len(rows),
        inserted_count=inserted_count,
        avg_utility=sum(utilities) / len(utilities),
        positive_precision=positives / len(utilities),
    )
