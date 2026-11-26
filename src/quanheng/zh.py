"""All user-facing Chinese copy lives in this module."""

PATCH_ANCHOR = "锚点不是假设追问的目标"

# ---- 档位 ----
LEVEL_NAMES = {
    "clear": "明确",
    "leaning": "倾向",
    "toss_up": "难分高下",
    "mostly_infeasible": "多数情况下没有可行方案",
}

# ---- 结论 ----
VERDICT_DECISIVE = "{level} {winner}　{winner} 在 {p} 的情况下胜出（{err}）"
VERDICT_TOSS_UP = "难分高下　{winner} 略占上风，在 {p} 的情况下胜出（{err}）"
VERDICT_INFEASIBLE = "多数情况下没有可行方案　{p} 的情况下没有任何候选满足全部硬约束"
REMINDER = "在你给出的范围和偏好之内成立，不代表事后一定更好。"
SOME_INFEASIBLE = "有 {p} 的情况下没有任何候选可行。"
OPTION_INFEASIBLE = "{option} 有 {p} 的情况不满足硬约束。"

# ---- 贡献 ----
CONTRIB_TITLE = "为什么是 {leader}（按中心值计算，{leader} 相对 {second} 的各项贡献）"
CONTRIB_LINE = "  {criterion} {diff}"
CENTRAL_DISAGREES = "注意：按中心值计算是 {option} 领先，与抽样结果不一致，以抽样结果为准。"

# ---- 翻盘点 ----
TIPPING_TITLE = "什么会让结论翻转"
TIPPING_SCORE = '  · {option} 的"{criterion}"{direction} {value}{unit}（你填的是 {stated}）'
TIPPING_WEIGHT = '  · "{criterion}"的权重{direction} {value}（现在是 {current}）'
TIPPING_REST = "  · 其余 {n} 个输入在参照范围内怎么变都不会翻转结论"
TIPPING_NONE = "  · 在参照范围内，任何单个输入怎么变都不会翻转结论"
BELOW = "低于"
ABOVE = "高于"

# ---- 信息 ----
INFO_TITLE = "最值得补的信息"
INFO_SCORE = (
    '  {n}. {option} 的"{criterion}"：在你给的范围低端时 {leader} 的胜出率是 {low}，高端时是 {high}'
)
INFO_WEIGHT = '  {n}. "{criterion}"有多重要：权重偏低时 {leader} 的胜出率是 {low}，偏高时是 {high}'
INFO_NONE = "  没有哪一个输入单独弄准后会明显改变把握。"

# ---- 注意 ----
OUTSIDE_ANCHORS = '注意：{option} 的"{criterion}"超出了你设的参照范围。'
POOR_CONSISTENCY_NOTE = "注意：两两比较的结果前后不够一致，权重的波动已放大。"

# ---- 约束差距 ----
GAP_TITLE = "放宽哪条约束最容易得到可行方案"
GAP_LINE = '  · "{criterion}" {op} {threshold}：最接近的是 {option}，还差 {gap}{unit}'

# ---- 假设 ----
ASSUMPTIONS_TITLE = "这次计算的假设"
ASSUMPTION_PREFIX = "  · "
ASSUMPTION_LINES = (
    "各个不确定的输入相互独立",
    '数值按"差"和"好"两个参照值线性换算成分数',
    "只给了范围的输入，范围内每个值同样可能",
    "给了最可能值的输入，越靠近最可能值越可能",
    "权重按你选的把握程度上下浮动",
)

# ---- 末行 ----
META_LINE = "抽样 {samples} 次，种子 {seed}，引擎 {engine}，NumPy {numpy}"

# ---- 数值写法 ----
RANGE_TEXT = "{low}–{high}"
THREE_POINT_TEXT = "{low}–{high}，最可能 {mode}"

# ---- 假设追问 ----
WHATIF_SCORE_TARGET = '{option} 的"{criterion}"'
WHATIF_WEIGHT_TARGET = '"{criterion}"的权重'
WHATIF_CONSTRAINT_TARGET = '"{criterion}"的门槛'
WHATIF_CHANGE = "{target}：{before} → {after}"
WHATIF_BEFORE = "之前：{verdict}"
WHATIF_AFTER = "之后：{verdict}"
WHATIF_FLIPPED = "结论翻转了。"
WHATIF_NOT_FLIPPED = "结论没有翻转。"
