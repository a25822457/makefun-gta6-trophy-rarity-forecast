"""gta6-trophy-rarity-forecast-01 entry in Python 3.13 (standard library only).

Forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
rate on Exophase at the case's target, as five quantiles (0.05, 0.25, 0.5, 0.75,
0.95) in percent. Reads every <id>.in in the input directory and writes
<id>.out to the output directory, one row per trophy:
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.

Method (per trophy, on the logit scale):
  1. Trend: the case's history is a step function -- Exophase's page only
     changes when it refreshes, so consecutive identical reads are collapsed
     to one point each. A recency-weighted least-squares line is fitted to
     logit(rate) against hours since release, weights halving every 24 h,
     and read at the target. The fit is on plain hours, not log-time: a
     rate's logit rises near-linearly in time while players work through
     the content, and log-time would compress the recent, fast growth and
     under-forecast. This is the data-driven leg; it already carries the
     dilution (players_tracked growing faster than earners pulls rates
     down) and the progression (earners catching up pulls them up).
  2. Prior: a power-law growth factor per trophy category, anchored on the
     trophy's last observed rate -- story trophies climb fast and level off
     high, grind / collectible / robbery trophies climb slowly and level off
     low, 100% completion and the platinum fall (dilution: almost nobody
     finishes a ~200 h checklist in week one, while casual players keep
     joining Exophase). The factor is calibrated on GTA V / Red Dead
     Redemption 2 week-one Exophase behaviour, and the rate is capped at
     the category ceiling, which is what stops a log-time trend from
     extrapolating a story trophy to 90%+.
  3. Blend: the forecast median is a weighted mean of the trend and the
     prior in logit space. The trend's weight grows with the number of
     distinct history points and the fit's R^2, and with short horizons the
     trend (persistence) is trusted more; over the final's 4.5-day horizon
     the prior carries real weight, which is what stops a log-time line
     from extrapolating a story trophy to 90%+.
  4. Spread: a random-walk term (0.5 logit units per sqrt-day, the house
     baseline's scale) plus the trend fit's residual spread plus a term for
     how far the trend and the prior disagree. The 5%/95% quantiles are the
     median shifted by +/-1.645 spreads in logit space, so a trophy whose
     history is noisy or whose trend and prior disagree gets an honestly
     wider band.

No network, no randomness (deterministic: same input bytes give the same
output bytes). Runs in well under the 8-second limit.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first (no sibling
# modules are needed, but this keeps the layout identical to the starter).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# GTA VI released 2026-11-19 (all times UTC). Hours are counted from here.
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- Trophy metadata (from the post-launch trophy guide's 55-trophy list,
# matched to the case's trophies by name; ids are Exophase order). ---
# id -> (kind, stage): kind is the trophy's nature, stage the roadmap stage
# (1 = early story, 2 = mid-game / 100% groundwork, 3 = late / cleanup,
# 4 = platinum).
TROPHY_META = {
    1: ("platinum", 4),
    2: ("story", 1),
    3: ("story", 1),
    4: ("story", 1),
    5: ("story", 1),
    6: ("story", 1),
    7: ("story", 1),
    8: ("story", 1),
    9: ("story", 1),
    10: ("story", 1),
    11: ("story", 1),
    12: ("story", 1),
    13: ("story", 1),
    14: ("completion", 2),
    15: ("activity", 2),
    16: ("activity", 3),
    17: ("activity", 2),
    18: ("activity", 2),
    19: ("activity", 2),
    20: ("activity", 3),
    21: ("activity", 3),
    22: ("story", 1),
    23: ("story", 1),
    24: ("story", 1),
    25: ("completion", 3),
    26: ("story", 1),
    27: ("story", 1),
    28: ("collectible", 1),
    29: ("story", 3),
    30: ("collectible", 2),
    31: ("collectible", 2),
    32: ("activity", 2),
    33: ("misc", 3),
    34: ("robbery", 3),
    35: ("exploration", 2),
    36: ("misc", 3),
    37: ("misc", 3),
    38: ("misc", 3),
    39: ("robbery", 3),
    40: ("robbery", 2),
    41: ("robbery", 3),
    42: ("exploration", 3),
    43: ("robbery", 2),
    44: ("misc", 3),
    45: ("misc", 3),
    46: ("misc", 3),
    47: ("misc", 3),
    48: ("misc", 3),
    49: ("misc", 3),
    50: ("misc", 3),
    51: ("misc", 3),
    52: ("economy", 2),
    53: ("economy", 3),
    54: ("economy", 3),
    55: ("economy", 3),
}

# --- Category priors for the rate's growth ---
# The prior is a power-law growth factor anchored on the trophy's last
# observed rate: prior_rate = last_rate * (T_target / T_last) ** gamma,
# capped at the category ceiling. gamma is the net exponent of the rate
# (earners growing minus players_tracked diluting): story trophies climb
# (players rush the story, earners outrun new joiners), side content
# climbs slowly, and 100% completion / the platinum fall (almost nobody
# finishes a ~200 h checklist in week one, while casual players keep
# joining Exophase and diluting the rate). Calibrated on GTA V / RDR2
# week-one Exophase behaviour; the ceilings are the week-one levels those
# games' equivalents level off near (Exophase's tracked players skew
# hardcore, so early story trophies run high).
# (kind, stage) -> (gamma, ceiling_percent)
CATEGORY_PRIOR = {
    ("story", 1): (0.90, 65.0),
    ("story", 3): (0.70, 30.0),
    ("activity", 2): (0.40, 20.0),
    ("activity", 3): (0.35, 15.0),
    ("collectible", 1): (0.40, 15.0),
    ("collectible", 2): (0.40, 15.0),
    ("robbery", 2): (0.35, 12.0),
    ("robbery", 3): (0.35, 10.0),
    ("exploration", 2): (0.50, 20.0),
    ("exploration", 3): (0.40, 15.0),
    ("economy", 2): (0.40, 18.0),
    ("economy", 3): (0.35, 15.0),
    ("misc", 3): (0.30, 12.0),
    ("completion", 2): (-0.10, 3.0),
    ("completion", 3): (-0.10, 2.5),
    ("platinum", 4): (-0.10, 0.5),
}
DEFAULT_PRIOR = (0.35, 12.0)  # for a (kind, stage) not listed above

# The trend's weight halves every 24 h of age (recency weighting).
TREND_HALF_LIFE_HOURS = 24.0
# The random-walk spread, in logit units per square-root day (house scale).
RW_SIGMA = 0.5


def clamp_pct(x: float) -> float:
    """Clamp a percent to the range the logit is defined on."""
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(dt: datetime) -> float:
    return (dt - RELEASE).total_seconds() / 3600.0


def distinct_points(history, idx):
    """Collapse the history to one point per distinct (rates, tracked) value.

    Exophase's page is a step function: it only changes on a refresh, so
    long flat stretches carry one observation, not many. Returns a list of
    (hours_since_release, rate_percent) in time order, keeping the FIRST
    time each distinct value appears (when that refresh's values start).
    Reads with a null rate for this trophy are skipped.
    """
    points = []
    prev_key = None
    for read in history:
        rate = read["rates"][idx]
        if rate is None:
            continue
        key = (rate, read["players_tracked"])
        if key == prev_key:
            continue
        prev_key = key
        points.append((hours_since_release(parse_time(read["at"])), float(rate)))
    return points


def weighted_line(points, t_last):
    """Recency-weighted least squares of logit(rate) on hours since release.

    A rate's logit rises roughly linearly in time while players are working
    through the content (a sigmoid in time is near-linear in logit space
    through its rising phase), so the fit is on plain hours, not log-time:
    log-time would compress the recent, fast growth and under-forecast the
    target. Returns (intercept, slope, resid_std, r2). With fewer than two
    distinct x values the slope is 0 and the line sits at the last rate.
    """
    xs = [max(0.0, t) for t, _ in points]
    ys = [logit(r) for _, r in points]
    ws = [0.5 ** ((t_last - t) / TREND_HALF_LIFE_HOURS) for t, _ in points]
    sw = sum(ws)
    if sw <= 0:
        return logit(points[-1][1]), 0.0, 0.0, 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        resid = math.sqrt(sum(w * (y - my) ** 2 for w, y in zip(ws, ys)) / sw)
        return my, 0.0, min(resid, 1.5), 0.0
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    syy = sum(w * (y - my) ** 2 for w, y in zip(ws, ys))
    resid = math.sqrt(max(0.0, sum(w * (y - (intercept + slope * x)) ** 2
                                   for w, x, y in zip(ws, xs, ys)) / sw))
    r2 = (sxy * sxy / (sxx * syy)) if syy > 1e-12 else 0.0
    return intercept, slope, min(resid, 1.5), max(0.0, min(1.0, r2))


def prior_rate(trophy_id, last_rate, t_last, t_target):
    """The category prior's rate (percent) at the target.

    A power-law growth factor anchored on the trophy's last observed rate,
    so the prior never contradicts what the case actually shows: a trophy
    already at 100% stays near 100% (floored at 90% of its last rate), and
    a trophy whose launch-night rate sits above its category ceiling is
    allowed to fall as casual players flood in. The rate is capped at the
    category ceiling, so a story trophy's prior cannot run away past ~65%.
    """
    kind, stage = TROPHY_META.get(trophy_id, ("misc", 3))
    gamma, ceiling = CATEGORY_PRIOR.get((kind, stage), DEFAULT_PRIOR)
    t0 = max(0.5, t_last)
    t1 = max(0.5, t_target)
    rate = last_rate * ((t1 / t0) ** gamma)
    rate = min(rate, ceiling)
    if last_rate >= 50.0:
        rate = max(rate, last_rate * 0.9)
    return clamp_pct(rate)


def forecast_trophy(trophy_id, points, last_rate, t_last, t_target, horizon_days):
    """Five quantiles (percent) for one trophy at the target."""
    intercept, slope, resid, r2 = weighted_line(points, t_last)
    trend_logit = intercept + slope * max(0.0, t_target)
    prior_logit = logit(prior_rate(trophy_id, last_rate, t_last, t_target))

    # Trend weight: more distinct points and a better fit earn more weight;
    # short horizons trust persistence (the trend) more than the prior.
    n = len(points)
    if n <= 1:
        w = 0.15
    elif n == 2:
        w = 0.30
    else:
        w = 0.35 + 0.05 * min(n, 9)
    w *= 0.5 + 0.5 * r2
    w += (1.0 - w) * 0.5 * math.exp(-horizon_days / 0.5)
    w = max(0.05, min(0.85, w))

    median_logit = w * trend_logit + (1.0 - w) * prior_logit
    median_logit = max(logit(0.01), min(logit(99.99), median_logit))

    # Spread: random walk over the horizon, the trend fit's residual, and
    # how far the trend and the prior disagree (disagreement is uncertainty).
    rw = RW_SIGMA * math.sqrt(max(0.0, horizon_days))
    disagree = abs(trend_logit - prior_logit)
    spread = math.sqrt(rw * rw + resid * resid + (0.25 * disagree) ** 2)
    spread = max(0.4, min(2.0, spread))

    qs = []
    for z in Z:
        q = inv_logit(median_logit + z * spread)
        qs.append(round4(clamp_pct(q)))
    # Guarantee non-decreasing after rounding.
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    target = parse_time(case["target"])
    t_target = hours_since_release(target)
    horizon_days = case["horizon_minutes"] / 1440.0
    last_read = case["history"][-1]
    t_last = hours_since_release(parse_time(last_read["at"]))
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        points = distinct_points(case["history"], i)
        last_rate = None
        for read in reversed(case["history"]):
            if read["rates"][i] is not None:
                last_rate = float(read["rates"][i])
                break
        if last_rate is None:
            last_rate = 0.01
        if not points:
            points = [(t_last, last_rate)]
        qs = forecast_trophy(trophy["id"], points, last_rate, t_last,
                             t_target, horizon_days)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
