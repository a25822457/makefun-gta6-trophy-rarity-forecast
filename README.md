# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/a25822457/makefun-gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/a25822457/makefun-gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles (0.05, 0.25, 0.5,
0.75, 0.95) in percent.

Built for the make.fun bounty
[Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## Program

`main.py` (Python 3.13, standard library only, no network) reads every
`<id>.in` in the input directory and writes `<id>.out` to the output
directory: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`,
one row per trophy, quantiles non-decreasing, each 0–100.

Run locally on the practice cases:

```
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Two runs write byte-identical output; the program runs in well under a
second per case.

## Method

Per trophy, on the logit scale, four legs:

1. **Trend.** Exophase's page is a step function -- it only changes on a
   refresh -- so consecutive identical reads collapse to one point each. A
   recency-weighted least-squares line (weights halving every 24 h) is
   fitted to logit(rate) against hours since release and read at the
   target. The fit is on plain hours, not log-time: a rate's logit rises
   near-linearly in time while players work through the content, and
   log-time would compress the recent, fast growth and under-forecast.
   This leg already carries the two forces in the data: dilution
   (players_tracked growing faster than earners pulls rates down) and
   progression (earners catching up pulls them up).
2. **Prior.** A power-law growth factor per trophy category, anchored on
   the trophy's last observed rate: story trophies climb fast (gamma 0.9)
   and level off high (ceiling 65%), side content (activities,
   collectibles, robberies, exploration, economy, misc) climbs slowly
   (gamma 0.1–0.5, ceilings 10–20%), and 100% completion and the platinum
   fall (gamma −0.1, ceilings 3% and 0.5%: almost nobody finishes a ~200 h
   checklist in week one, while casual players keep joining Exophase and
   diluting the rate). Calibrated on GTA V / Red Dead Redemption 2 week-one
   Exophase behaviour. The rate is capped at the category ceiling, which
   is what stops a linear-time trend from extrapolating a story trophy
   past ~65%. A trophy already at 100% stays near 100% (floored at 90% of
   its last rate).
3. **Blend.** The forecast median is a weighted mean of the trend and the
   prior in logit space. The trend's weight grows with the number of
   distinct history points and the fit's R², and with short horizons the
   trend (persistence) is trusted more; over the final's 4.5-day horizon
   the prior carries real weight.
4. **Spread.** A random-walk term (0.5 logit units per sqrt-day, the house
   baseline's scale) plus the trend fit's residual spread plus a term for
   how far the trend and the prior disagree. A trophy whose history is
   noisy or whose trend and prior disagree gets an honestly wider band.

## Validation

- Practice cases p00/p01: valid output, byte-identical across two runs.
  Medians track the last read with small category adjustments (platinum
  0.63% → 0.56%, the 100% trophy stays ~99.6%, story trophies 1.1% →
  ~1.3–1.6%).
- A synthetic final-like case (120 reads over 2.7 days, 11 refreshes,
  tracked 638 → ~112,000, target 4.5 days out, invented "true" week-one
  rates per category): mean pinball loss 0.32 vs 0.84 for the house
  persistence baseline and 0.35 for the house log-time curve baseline on
  the same case, with 90% coverage 1.00. The synthetic truth is invented,
  so this is a sanity check, not a claim about the real final.

## Trophy data

The 55 trophies' categories (kind/stage) come from the post-launch trophy
guide's list, matched to the case's trophies by name; the ids are
Exophase order. No guide text is shipped here -- only the category per
trophy id, baked into `main.py` as `TROPHY_META`.
