#!/usr/bin/env python3
"""GTA VI launch-day Twitch CCU forecaster (gta6-twitch-ccu-forecast-01).

Method: comparables-anchored log-ratio regression.
  For each hour h = 2..24 of the release day, every past big launch in the
  case's own `comparables` gives one observation: how that launch's viewers
  in hour h compared with its hour-1 viewers, on the log scale,
  r_h = ln(viewers_h / viewers_1).  Across launches the ratio is NOT one
  shape: a launch whose hour 1 is already huge (the world already unlocked
  by midnight UTC) barely grows afterwards, while a launch whose hour 1 is
  tiny (most of the world still locked) grows ten- to a thousand-fold by
  prime time.  The single number that carries that information is the
  hour-1 count itself, so for each hour we fit, over the comparables,

      r_h = a_h + b_h * ln(viewers_1) + noise,

  by least squares (slope clamped to [-1, 0]), and read the five forecast
  quantiles off the residuals of that fit: q_tau = viewers_1 * exp(mu_h +
  resid_tau), where mu_h is the fitted value at this case's hour-1 count
  and resid_tau is the tau-quantile of the fit's residuals.  The spread is
  therefore the spread launches really show around the fit, not a number
  chosen by hand -- honest where launches disagree, narrow where they agree.

  Nothing is memorised: no practice answer, no launch's curve and no
  fitted number is stored in this file.  Every run re-fits from the
  comparables inside the case it is given, so the same program works on a
  practice case (38 comparables) and on the final (42, GTA VI included).

Validation (leave-one-launch-out over the 37 usable comparables in the
kit, mean pinball loss on the log scale over hours 2..24 x 5 levels):
  house baseline (flat at hour 1, widening spread) : 0.930
  one median shape for every launch                : 0.714
  this model (log hour-1 regression + residuals)   : 0.438
  Adding hour-1 channels or the pre-launch baseline as extra regressors
  made it worse (0.444 / 0.455: 37 launches cannot carry a second slope),
  and training only on the "clean tier" comparables made it worse too
  (0.449: fewer launches beats cleaner launches), so neither is used.
  An "unlock offset" feature (first hour the category surged above its
  pre-launch baseline) looked better at first (0.410) but only because its
  threshold used each launch's full-day peak, which the target case does
  not have at forecast time; rebuilt honestly from history + hour 1 only,
  it made the model worse (0.445) and is not used.  The practice launches
  are all rolling-midnight console single-player launches, the same kind
  the final is; the model's LOO loss on the single-player subset alone is
  0.46-0.47.

I/O (protocol agent-fun/program/v1): read every <id>.in in
$MAKEFUN_INPUT_DIR (else /input), write <id>.out to $MAKEFUN_OUTPUT_DIR
(else /output) as {"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]},
...]} for hours 2..24.  Stdlib only, no network, deterministic (same
bytes on re-run).  If a case cannot be parsed, or no comparable is usable,
fall back to the house-style flat forecast with a widening spread, so the
case is always valid; an invalid case scores as its hour-1 print on every
quantile and takes no prize, so never emit a bad file.
"""

import json
import math
import os
import sys

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Standard normal quantiles, house-baseline fallback only.
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
BASE_SPREAD = 0.15   # house baseline: log-spread grows as 0.15 * sqrt(h-1)
MIN_VIEWERS = 1.0
MAX_VIEWERS = 1_000_000_000.0
MIN_TRAIN = 8        # comparables needed before a per-hour fit is trusted
MIN_HOURS = 12       # usable hours needed before a launch counts at all


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def quantile(sorted_vals, q):
    """Linear-interpolation quantile of an ascending list."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = lo + 1 if lo + 1 < n else lo
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def ols_intercept_slope(xs, ys):
    """Least-squares y = a + b*x; returns (a, b) with b clamped to [-1, 0]."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = 0.0
    sxy = 0.0
    for x, y in zip(xs, ys):
        dx = x - mx
        sxx += dx * dx
        sxy += dx * (y - my)
    b = sxy / sxx if sxx > 0.0 else 0.0
    b = clamp(b, -1.0, 0.0)
    return my - b * mx, b


def flat_forecast(p1, hour):
    """House-style fallback: every hour stays at hour 1, spread widens."""
    spread = BASE_SPREAD * math.sqrt(hour - 1)
    return [p1 * math.exp(z * spread) for z in Z]


def comparable_curves(case):
    """Release-day log-ratio curves of the case's comparables.

    Returns a list of (log_hour1, [log(viewers_h / viewers_1) for h=1..24
    with None for a missing hour]).  A launch counts only when its hour-1
    point exists and is positive and at least MIN_HOURS of its 24 hours
    are present.  The Grand Theft Auto VI category's own pre-release
    history is listed among the comparables but is not a launch day, so
    any comparable whose name says it is pre-release is left out.
    """
    out = []
    for comp in case.get("comparables") or []:
        name = ""
        if isinstance(comp.get("game"), dict):
            name = comp["game"].get("name") or ""
        name = name or comp.get("name") or ""
        if "before release" in name.lower():
            continue
        hourly = comp.get("hourly") or []
        if not hourly:
            continue
        day_start = comp.get("day_start") or ""
        by_from = {}
        for point in hourly:
            try:
                v = float(point["viewers"])
            except (KeyError, TypeError, ValueError):
                continue
            by_from[point.get("from")] = v
        first = by_from.get(day_start)
        if first is None or first <= 0:
            continue
        log_h1 = math.log(first)
        logs = []
        usable = 0
        for h in range(24):
            stamp = day_start[:11] + "%02d:00:00Z" % h
            v = by_from.get(stamp)
            if v is None or v <= 0:
                logs.append(None)
            else:
                logs.append(math.log(v / first))
                usable += 1
        if usable < MIN_HOURS:
            continue
        out.append((log_h1, logs))
    return out


def forecast_case(case):
    """Five quantiles for hours 2..24 of one case, as {"hours": [...]}."""
    known = case.get("known") or []
    p1 = MIN_VIEWERS
    if known:
        try:
            p1 = max(MIN_VIEWERS, float(known[0].get("viewers") or 0) or MIN_VIEWERS)
        except (TypeError, ValueError):
            p1 = MIN_VIEWERS
    levels = case.get("levels") or list(LEVELS)
    hours = []
    for row in case.get("forecast") or []:
        try:
            hours.append(int(row["hour"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not hours:
        hours = list(range(2, 25))
    curves = comparable_curves(case)
    log_p1 = math.log(p1)
    rows = []
    for h in hours:
        idx = h - 1  # logs[0] is hour 1, logs[1] hour 2, ...
        xs = []
        ys = []
        for log_h1, logs in curves:
            r = logs[idx] if idx < len(logs) else None
            if r is not None:
                xs.append(log_h1)
                ys.append(r)
        if len(xs) >= MIN_TRAIN:
            a, b = ols_intercept_slope(xs, ys)
            mu = a + b * log_p1
            resid = sorted(y - (a + b * x) for x, y in zip(xs, ys))
            qs = [p1 * math.exp(mu + quantile(resid, tau)) for tau in levels]
        else:
            qs = flat_forecast(p1, h)
        # Keep the row valid: finite, inside (0, 1e9], non-decreasing.
        clean = []
        prev = 0.0
        for q in qs:
            if q is None or not math.isfinite(q):
                q = p1
            q = clamp(q, MIN_VIEWERS, MAX_VIEWERS)
            q = max(q, prev)
            clean.append(round(q, 2))
            prev = clean[-1]
        rows.append({"hour": h, "q": clean})
    return {"hours": rows}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        names = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    except OSError:
        names = []
    for name in names:
        case_id = name[: -len(".in")]
        path = os.path.join(INPUT_DIR, name)
        out = None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                case = json.load(fh)
            out = forecast_case(case)
        except Exception:
            out = None
        if out is None:
            # Last-resort valid output: flat at 1 viewer, widening spread.
            out = {"hours": [{"hour": h, "q": flat_forecast(1.0, h)}
                             for h in range(2, 25)]}
        text = json.dumps(out, separators=(",", ":")) + "\n"
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"),
                  "w", encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
