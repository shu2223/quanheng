from typing import Literal

from pydantic import Field

from quanheng._base import Model


class Thresholds(Model):
    clear: float = 0.90
    leaning: float = 0.70


class Meta(Model):
    seed: int
    samples: int
    engine_version: str
    numpy_version: str


class Verdict(Model):
    level: Literal["clear", "leaning", "toss_up", "mostly_infeasible"]
    winner: str | None
    runner_up: str | None
    win_rate: float
    std_error: float


class Margin(Model):
    p5: float
    p50: float
    p95: float
    used_share: float
    bin_edges: list[float]
    counts: list[int]


class ConstraintGap(Model):
    index: int
    criterion: str
    nearest_option: str
    gap: float


class ResultWarning(Model):
    code: Literal["outside_anchors", "central_disagrees", "poor_consistency"]
    target: str | None = None


class TippingPoint(Model):
    target: str
    current: float
    flips_at: float | None
    new_leader: str | None
    within_stated_range: bool


class InfoValue(Model):
    target: str
    share: float
    leader_win_rate_low: float
    leader_win_rate_high: float


class ChoiceResult(Model):
    kind: Literal["choice_result"] = "choice_result"
    meta: Meta
    verdict: Verdict
    win_rates: dict[str, float]
    none_feasible_share: float
    feasibility: dict[str, float]
    central_scores: dict[str, float]
    contributions: dict[str, dict[str, float]]
    central_leader: str | None
    margin: Margin | None
    tipping_points: list[TippingPoint] = Field(default_factory=list)
    info_values: list[InfoValue] = Field(default_factory=list)
    constraint_gaps: list[ConstraintGap]
    warnings: list[ResultWarning]
    assumptions: list[str]
