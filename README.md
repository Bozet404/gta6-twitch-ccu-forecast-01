# GTA VI launch-day Twitch CCU forecaster

Entry for `gta6-twitch-ccu-forecast-01` (make.fun, $250, top 20 paid).
A program that reads one case and forecasts, for hours 2–24 of GTA VI's
release day (UTC), five quantiles (5%, 25%, 50%, 75%, 95%) of Twitch
viewers in the GTA VI category. Free entry, stdlib-only Python, offline,
deterministic, ~0.2 s per case.

## How it works

For each hour h = 2..24, every past big launch in the case's own
`comparables` gives one observation: how that launch's viewers in hour h
compared with its hour-1 viewers, on the log scale,
`r_h = ln(viewers_h / viewers_1)`.

The ratio is not one shape. A launch whose hour 1 is already huge (the
world already unlocked by midnight UTC) barely grows afterwards, while a
launch whose hour 1 is tiny (most of the world still locked) grows ten-
to a thousand-fold by prime time. The hour-1 count itself carries that
information, so for each hour the program fits, over the comparables,

```
r_h = a_h + b_h * ln(viewers_1) + noise
```

by least squares (slope clamped to [-1, 0]) and reads the five forecast
quantiles off the residuals of that fit:

```
q_tau = viewers_1 * exp(mu_h + resid_tau)
```

where `mu_h` is the fitted value at this case's hour-1 count and
`resid_tau` is the tau-quantile of the fit's residuals. The spread is
therefore the spread launches really show around the fit — honest where
launches disagree, narrow where they agree.

Nothing is memorised: no practice answer, no launch's curve and no fitted
number is stored here. Every run re-fits from the comparables inside the
case it is given, so the same program works on a practice case (38
comparables) and on the final (42, GTA VI included). The GTA VI category's
own pre-release history is listed among the comparables but is not a
launch day, so it is left out of the fit.

## Validation

Leave-one-launch-out over the 37 usable comparables in the kit, mean
pinball loss on the log scale (hours 2..24 × 5 levels):

| model | LOO pinball loss |
|---|---:|
| house baseline (flat at hour 1, widening spread) | 0.930 |
| one median shape for every launch | 0.714 |
| **this model (log hour-1 regression + residuals)** | **0.438** |

Rejected by the same test (all worse than 0.438): adding hour-1 channels
(0.444), adding the pre-launch baseline (0.455), training only on the
"clean tier" comparables (0.449), and an unlock-offset feature rebuilt
honestly from history + hour 1 only (0.445). An unlock-offset variant
that used each launch's full-day peak looked better (0.410) but leaks
information the target case does not have at forecast time, so it was
dropped.

Local checks on the 4 practice cases: outputs are valid (23 rows, hours
2–24, 5 ordered quantiles each, all in (0, 1e9], ~1.5 KB per file),
byte-identical on re-run, 0.19 s total. There is no local grade for this
competition — the sealed scorer and reference data live on the server, so
the true practice loss is only known after submission.

## Files

- `main.py` — the forecaster (the entry point).
- `.makefun/owner.json` — payout wallet for the entry.
- `LICENSE` — MIT.

## Run

```
MAKEFUN_INPUT_DIR=<dir with .in files> MAKEFUN_OUTPUT_DIR=<dir for .out files> python3 main.py
```

Defaults: `/input` and `/output`.
