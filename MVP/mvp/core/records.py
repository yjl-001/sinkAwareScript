from dataclasses import dataclass


@dataclass
class CandidateRecord:
    """一个 delimiter 候选点的观测与反事实结果。

    这里的 step 表示“生成第 step 个 completion token 之前”的位置。
    如果当前前缀最后一个真实 token 是 delimiter，就可以在这个 step 前
    插入 inference latent memory。
    """

    sample_idx: int
    reference_mode: str
    reference_reward: float
    candidate_rank: int
    step: int
    generated_so_far: int
    rel_pos: float
    pos_bucket: int
    delimiter_token_id: int
    delimiter_text: str
    sink_mass: float
    sink_mass_z: float
    # first_key_attention 是“当前 query 对第一个有效 key/token 的 attention”，
    # 通常按最后 N 层和所有 heads 平均。它和 sink_mass 不同：sink_mass 可以看
    # 前多个 key，而这里严格只看第一个 key。
    first_key_attention: float
    entropy: float
    entropy_z: float
    # branch_reward/utility 只在 single-insertion branch 中填充：
    # utility = reward(只在该点插一次 latent) - reward(prompt-only baseline)。
    branch_reward: float | None = None
    utility: float | None = None
    # single-insertion branch 里真实发生的插入位置。正常应为 [step]；
    # 如果为空，说明分支在该 step 前结束，或该 step 已不再是 delimiter 后位置。
    branch_forced_steps_used: list[int] | None = None
    branch_inserted: bool | None = None


@dataclass
class StrategyRecord:
    """一个完整 rollout 策略在单个样本上的结果。"""

    sample_idx: int
    reference_mode: str
    strategy: str
    reward: float
    # planned_steps 是策略根据 baseline candidates 计划插入的位置。
    planned_steps: list[int]
    # inserted_steps 是 rollout 中真正发生插入的位置；如果轨迹偏离、提前 EOS，
    # 或该 step 不再是 delimiter 后位置，计划点会被跳过。
    inserted_steps: list[int]
    completion: str


@dataclass
class SequencePointRecord:
    """baseline 轨迹上每一个可观测 step 的 sink score。

    和 CandidateRecord 不同，它不要求当前位置在 delimiter 后面，因此可以用于
    “全序列 sink 阈值触发”这类实验。
    """

    sample_idx: int
    reference_mode: str
    point_rank: int
    step: int
    generated_so_far: int
    rel_pos: float
    pos_bucket: int
    token_id: int
    token_text: str
    prefix_ends_with_delimiter: bool
    first_key_attention: float


@dataclass
class SelectedPointRecord:
    """某个实验组真正选择并评估的单点反事实结果。"""

    sample_idx: int
    reference_mode: str
    group: str
    source: str
    step: int
    rank_in_group: int
    score_name: str
    score: float
    requires_delimiter: bool
    reference_reward: float
    branch_reward: float | None = None
    utility: float | None = None
    branch_forced_steps_used: list[int] | None = None
    branch_inserted: bool | None = None


@dataclass
class GroupResultRecord:
    """一个 sample 上一个实验组的聚合结果。"""

    sample_idx: int
    reference_mode: str
    group: str
    selected_count: int
    inserted_count: int
    avg_utility: float
    positive_precision: float


@dataclass
class GenerationTrace:
    """一次生成的完整返回值。

    completion_ids/completion 用于 reward；candidates 只在 baseline
    collect_candidates=True 时有值；forced_steps_used 记录真实插入的位置。
    """

    completion_ids: list[int]
    completion: str
    candidates: list[CandidateRecord]
    sequence_points: list[SequencePointRecord]
    forced_steps_used: list[int]
