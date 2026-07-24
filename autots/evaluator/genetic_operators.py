"""Genetic operators for the AutoTS template search.

Standalone building blocks used by NewGeneticTemplate: local numeric mutation
of parameter dicts, annealed parent-selection weights, and an optional
surrogate model that ranks candidate templates before they are evaluated.

This module must not import from autots.evaluator.auto_model or model code
(they import it); everything operates on plain dicts and template DataFrames.
Randomness uses the module-global random/np.random state AutoTS seeds.
"""

import copy
import json
import random
import zlib

import numpy as np
import pandas as pd

GENETIC_PARAMS_DEFAULTS = {
    'mutation': True,
    'mutation_probability': 0.3,
    'anneal': True,
    'surrogate': False,
    'surrogate_oversample': 3,
    'surrogate_fraction': 0.8,
    'surrogate_min_rows': 50,
    'surrogate_max_family_fraction': 0.35,
    'ensemble_rank_sampled': 8,
    'ensemble_rank_pool': 25,
}


def resolve_genetic_params(genetic_params=None):
    """Fill missing genetic search options with defaults. Idempotent.

    Args:
        genetic_params (dict): any subset of GENETIC_PARAMS_DEFAULTS keys

    Raises:
        ValueError: on unrecognized keys, so typos fail at AutoTS() time
    """
    if genetic_params is None:
        return dict(GENETIC_PARAMS_DEFAULTS)
    unknown = set(genetic_params) - set(GENETIC_PARAMS_DEFAULTS)
    if unknown:
        raise ValueError(
            f"genetic_params got unknown keys {sorted(unknown)}; "
            f"valid keys are {sorted(GENETIC_PARAMS_DEFAULTS)}"
        )
    return {**GENETIC_PARAMS_DEFAULTS, **genetic_params}


def mutate_numeric_value(value, intensity: float = 1.0, minimum=None, maximum=None):
    """Return a small local jitter of an int or float, preserving its type.

    Callers must exclude bools (bool is a subclass of int). Non-negative
    values stay non-negative, and floats starting in (0, 1] stay in (0, 1]
    as those are usually rates/ratios with hard upper bounds.

    When minimum/maximum are given the result is clamped to that inclusive
    range, so a value cannot drift outside the sanctioned bounds a caller
    derived from the model's own parameter sampler (see perturb_params).
    """
    factors = [
        1 - 0.3 * intensity,
        1 - 0.2 * intensity,
        1 + 0.25 * intensity,
        1 + 0.4 * intensity,
    ]
    if isinstance(value, int):
        if abs(value) <= 3:
            # small ints are usually enumerated orders/lags where a single
            # step is the meaningful neighborhood
            new_value = value + random.choice([-1, 1])
        else:
            new_value = int(round(value * random.choice(factors)))
            if new_value == value:
                new_value = value + random.choice([-1, 1])
        if value >= 0 and new_value < 0:
            new_value = value + 1
        if minimum is not None:
            new_value = max(new_value, int(np.ceil(minimum)))
        if maximum is not None:
            new_value = min(new_value, int(np.floor(maximum)))
        # a sub-integer sanctioned range (ceil(min) > floor(max)) holds no
        # valid integer; leave the value untouched rather than exit the range
        if minimum is not None and new_value < minimum:
            return int(value)
        if maximum is not None and new_value > maximum:
            return int(value)
        return int(new_value)
    elif isinstance(value, float):
        if value == 0.0:
            # 0.0 is often an 'off' sentinel, and jitter is multiplicative
            return value
        new_value = float(value * random.choice(factors))
        if 0.0 < value <= 1.0:
            new_value = min(new_value, 1.0)
        if minimum is not None:
            new_value = max(new_value, float(minimum))
        if maximum is not None:
            new_value = min(new_value, float(maximum))
        return new_value
    else:
        return value


def perturb_params(
    params,
    probability: float = 0.3,
    intensity: float = 1.0,
    max_depth: int = 6,
    bounds=None,
    _depth: int = 0,
    _path: str = "",
):
    """Locally mutate numeric leaves of a model or transformation param dict.

    Structure is preserved exactly: bools, Nones, strings (categoricals and
    coupled 'model' selectors), and lists/tuples are left untouched, so the
    output always has the same shape and key set as the input, only with
    some int/float values nudged. The input dict is never modified.

    Args:
        params (dict): parameters as from get_new_params or json.loads
        probability (float): chance each numeric leaf is jittered
        intensity (float): scales jitter magnitude
        max_depth (int): recursion bound for deeply nested param dicts
        bounds (dict): optional {dotted-path: (minimum, maximum)} confining
            each numeric leaf to the range its model's own sampler produces
            (see numeric_bounds_from_samples). When given, ONLY leaves whose
            path was observed are mutated, and each is clamped into its range.
            Cost-driving integers (n_estimators, max_windows, ...) appear in
            every draw with a wide range, so they are always clamped down to
            the sampled maximum - a surviving lineage can never drift them to
            arbitrarily slow values. (A value already outside the observed
            range, which only happens for a rarely-sampled conditional
            parameter, is left unchanged rather than pushed further out.)
            When None, every numeric leaf is eligible and jitter is unbounded
            - used by direct callers/tests, not the generational search.
    """
    if not isinstance(params, dict):
        return copy.deepcopy(params)
    mutated = copy.deepcopy(params) if _depth == 0 else params
    if _depth > max_depth:
        return mutated
    for key, value in mutated.items():
        child_path = f"{_path}.{key}" if _path else str(key)
        # bool must be tested before int; None and strings are categorical
        if isinstance(value, bool) or value is None or isinstance(value, str):
            continue
        elif isinstance(value, dict):
            # recurses into nested blocks like model_params while the sibling
            # 'model' selector, being a string, stays fixed alongside
            perturb_params(
                value, probability, intensity, max_depth, bounds,
                _depth + 1, child_path,
            )
        elif isinstance(value, (list, tuple)):
            # opaque: lists can mix scalars and tuples with structural meaning
            continue
        elif isinstance(value, (int, float)):
            if bounds is not None and child_path not in bounds:
                continue  # unsanctioned path, never seen from the sampler
            if random.random() < probability:
                minimum = maximum = None
                if bounds is not None:
                    minimum, maximum = bounds[child_path]
                mutated[key] = mutate_numeric_value(
                    value, intensity=intensity, minimum=minimum, maximum=maximum
                )
    return mutated


def _walk_numeric_leaves(obj, path, out, depth=0, max_depth=6):
    """Collect (dotted-path, value) for each numeric non-bool leaf, using the
    same traversal and path convention as perturb_params so bounds keys line
    up with the leaves they constrain."""
    if depth > max_depth:
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{path}.{key}" if path else str(key)
            _walk_numeric_leaves(value, child, out, depth + 1, max_depth)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        # bools, None, strings, and opaque lists/tuples are not numeric leaves
        out.append((path, obj))


def numeric_bounds_from_samples(sampler_fn, n_samples: int = 8, max_depth: int = 6):
    """Observe the numeric range each parameter path takes across repeated
    sampler draws, returning {path: (minimum, maximum)}.

    Passed to perturb_params so mutation of a surviving lineage stays inside
    the value range the model's own get_new_params would produce - the fix
    for cost-driving integers (n_estimators, max_windows, ...) otherwise
    drifting arbitrarily far over many generations.
    """
    bounds = {}
    for _ in range(n_samples):
        try:
            sample = sampler_fn()
        except Exception:
            continue
        leaves = []
        _walk_numeric_leaves(sample, "", leaves, 0, max_depth)
        for path, value in leaves:
            if path in bounds:
                low, high = bounds[path]
                bounds[path] = (min(low, value), max(high, value))
            else:
                bounds[path] = (value, value)
    return bounds


def transformation_numeric_bounds(trans_dict, n_samples: int = 6):
    """Build perturb_params bounds for one transformation-pipeline dict: each
    step's numeric params are bounded by that transformer's own
    get_transformer_params range, keyed by positional index to match
    transformation_params.<idx>.<param> paths (cached per transformer name).
    """
    from autots.tools.transform import get_transformer_params

    names = trans_dict.get('transformations', {}) or {}
    per_name = {}
    bounds = {}
    for idx, name in names.items():
        if name not in per_name:
            name_bounds = numeric_bounds_from_samples(
                lambda n=name: get_transformer_params(n), n_samples=n_samples
            )
            per_name[name] = name_bounds
        for path, rng in per_name[name].items():
            bounds[f"transformation_params.{idx}.{path}"] = rng
    return bounds


def compute_generation_progress(
    current_generation,
    max_generations,
    passed_time_minutes,
    generation_timeout_minutes,
):
    """Fraction of the generation search budget consumed, in [0, 1].

    AutoTS uses sentinel values for 'unbounded': max_generations becomes
    99999 when the run is timeout-driven and generation_timeout becomes 9e6
    minutes when generation-driven. Only real budgets count toward progress;
    if neither is real, a neutral 0.5 is returned.
    """
    ratios = []
    if max_generations and max_generations < 9000:
        ratios.append(current_generation / max_generations)
    if generation_timeout_minutes and generation_timeout_minutes < 8e6:
        ratios.append(passed_time_minutes / generation_timeout_minutes)
    if not ratios:
        return 0.5
    return float(min(max(max(ratios), 0.0), 1.0))


def annealed_parent_weights(kind: str, progress):
    """Parent-pool selection weights favoring fresh random draws early in the
    search and elite parents late, normalized to sum to 1.

    Returns None when progress is None, which callers treat as the exact
    legacy uniform-selection path.

    The orders are contracts with the parent arrays built in auto_model.py:
        'model_recombination': [fir, sec, r2, r]
        'transformer': [first_transformer, sec, best, r, r2]
        'keep_fresh_mutate': [current, fresh, mutant]
        'keep_fresh': [current, fresh]
    """
    if progress is None:
        return None
    p = min(max(float(progress), 0.0), 1.0)
    elite = 1.0 + p
    semi = 1.0 + 0.5 * p
    fresh = 2.0 - p
    mid = 1.0
    table = {
        'model_recombination': [elite, semi, fresh, mid],
        'transformer': [elite, semi, semi, fresh, fresh],
        'keep_fresh_mutate': [elite, fresh, mid],
        'keep_fresh': [elite, fresh],
    }
    weights = table[kind]
    total = sum(weights)
    return [x / total for x in weights]


def _flatten_param_tokens(obj, path=""):
    """Flatten params into (categorical tokens, numeric (path, value) pairs).

    Keys are sorted with key=str because JSON round-trips leave sibling dicts
    with mixed int and str keys ({0: ...} vs {"0": ...}).
    """
    tokens = []
    numerics = []
    if isinstance(obj, dict):
        for key in sorted(obj.keys(), key=str):
            sub_path = f"{path}.{key}" if path else str(key)
            sub_tokens, sub_numerics = _flatten_param_tokens(obj[key], sub_path)
            tokens.extend(sub_tokens)
            numerics.extend(sub_numerics)
    elif isinstance(obj, bool) or obj is None or isinstance(obj, str):
        tokens.append(f"{path}={obj}")
    elif isinstance(obj, (int, float)):
        value = float(obj)
        if np.isfinite(value):
            numerics.append((path, float(np.sign(value) * np.log1p(abs(value)))))
    else:
        # lists/tuples and anything exotic hash as one categorical value
        tokens.append(f"{path}={obj}")
    return tokens, numerics


def featurize_template_rows(template_df, n_features: int = 4096):
    """Hash Model name + both param JSONs into a dense float32 matrix.

    Categorical leaves count occurrences at crc32(token) dims; numeric leaves
    add sign(x)*log1p(|x|) at crc32(path#num) dims. zlib.crc32 rather than
    builtin hash() because the latter is salted per process, which would
    break run-to-run reproducibility.
    """
    X = np.zeros((template_df.shape[0], n_features), dtype=np.float32)
    rows = zip(
        template_df['Model'].astype(str),
        template_df['ModelParameters'].astype(str),
        template_df['TransformationParameters'].astype(str),
    )
    for i, (model, model_params, trans_params) in enumerate(rows):
        tokens = [f"Model={model}"]
        numerics = []
        for prefix, param_json in (("m", model_params), ("t", trans_params)):
            try:
                parsed = json.loads(param_json)
            except Exception:
                tokens.append(f"{prefix}#unparsable")
                continue
            sub_tokens, sub_numerics = _flatten_param_tokens(parsed, prefix)
            tokens.extend(sub_tokens)
            numerics.extend(sub_numerics)
        for token in tokens:
            X[i, zlib.crc32(token.encode('utf-8')) % n_features] += 1.0
        for num_path, value in numerics:
            X[i, zlib.crc32((num_path + "#num").encode('utf-8')) % n_features] += value
    return X


def _capped_rank_selection(order, models, k: int, cap: int, counts=None):
    """Greedily take up to k positions from best-first `order`, keeping at
    most `cap` per distinct value of `models[position]`.

    A surrogate that ranks purely by predicted score tends to flood its
    keep-list with whichever cheap, reliable model family it is most
    confident about (e.g. naive baselines), starving the search of diversity.
    This caps that without removing the ranking signal. If too few families
    exist to fill k under the cap, the best of the capped-out remainder
    backfills, so it always returns min(k, len(order)) selections.

    Args:
        counts (dict): running per-model tally to continue from; mutated in
            place and returned, so a second selection stage (e.g. the
            surrogate's random exploration fill) can share one cap budget
            with this one instead of each independently allowing up to cap.

    Returns (selected, deferred, counts): selected/deferred are arrays of
    positions into `order`'s index space, each preserving best-first
    relative order.
    """
    selected = []
    deferred = []
    if counts is None:
        counts = {}
    for idx in order:
        model = models[idx]
        if len(selected) < k and counts.get(model, 0) < cap:
            selected.append(idx)
            counts[model] = counts.get(model, 0) + 1
        else:
            deferred.append(idx)
    if len(selected) < k and deferred:
        need = k - len(selected)
        for idx in deferred[:need]:
            counts[models[idx]] = counts.get(models[idx], 0) + 1
        selected.extend(deferred[:need])
        deferred = deferred[need:]
    return np.array(selected, dtype=int), np.array(deferred, dtype=int), counts


def _capped_weighted_sample(pool, n: int, cap: int, counts: dict):
    """Weighted-random sample of n rows from pool (priority order, best
    first), skipping rows whose Model has reached cap in the running counts
    tally shared with _capped_rank_selection. Sampled one at a time (fixed
    log-rank weights) so the cap is enforced between draws; falls back to
    sampling without the cap once no eligible family remains, so it still
    returns min(n, len(pool)) rows.
    """
    n = min(n, pool.shape[0])
    if n <= 0:
        return pool.iloc[0:0]
    weight_of = dict(
        zip(pool.index, np.log(np.arange(pool.shape[0]) + 2)[::-1] + 1)
    )
    remaining = pool
    chosen = []
    for _ in range(n):
        eligible = remaining[
            remaining['Model'].map(lambda m: counts.get(m, 0) < cap)
        ]
        candidates = eligible if not eligible.empty else remaining
        weights = [weight_of[i] for i in candidates.index]
        pick = candidates.sample(1, weights=weights)
        chosen.append(pick)
        model = pick['Model'].iloc[0]
        counts[model] = counts.get(model, 0) + 1
        remaining = remaining.drop(pick.index)
    return pd.concat(chosen, axis=0)


def surrogate_select_candidates(
    new_template,
    model_results,
    max_results: int,
    keep_fraction: float = 0.8,
    min_training_rows: int = 50,
    max_family_fraction: float = 0.35,
    n_features: int = 4096,
    max_training_rows: int = 3000,
    verbose: int = 0,
):
    """Choose max_results rows from new_template using a surrogate ranker.

    Trains a fresh RandomForest each call on all results so far; Score is a
    relative snapshot recomputed as results accumulate, so ranks are only
    comparable within one call and the model is deliberately not reused.
    Failed models are trained on with the worst rank so the surrogate learns
    to avoid crash-prone parameter regions.

    No single Model may take more than max_family_fraction of max_results
    across the combined predicted-best and random-exploration selections, so
    the surrogate cannot collapse the search onto a single dominant family
    (see _capped_rank_selection / _capped_weighted_sample).

    Returns a DataFrame of selected candidates, or None when there is not
    yet enough history or anything fails - callers then fall back to the
    legacy weighted sampling.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor

        history = model_results[model_results['Ensemble'] == 0]
        history = history.drop_duplicates(
            subset=['Model', 'ModelParameters', 'TransformationParameters'],
            keep='last',
        )
        if history.shape[0] > max_training_rows:
            history = history.tail(max_training_rows)
        succeeded = history['Exceptions'].isna() & history['Score'].notna()
        if history.shape[0] < min_training_rows or succeeded.sum() < 25:
            return None
        y = pd.Series(1.0, index=history.index)
        y[succeeded] = history.loc[succeeded, 'Score'].rank(pct=True)
        regressor = RandomForestRegressor(
            n_estimators=50,
            min_samples_leaf=2,
            n_jobs=1,
            random_state=random.randint(0, 2**31 - 1),
        )
        regressor.fit(featurize_template_rows(history, n_features), y.to_numpy())
        predictions = regressor.predict(
            featurize_template_rows(new_template, n_features)
        )

        n_keep = int(max_results * keep_fraction)
        n_random = max_results - n_keep
        order = np.argsort(predictions, kind='stable')
        models_arr = new_template['Model'].to_numpy()
        cap = max(1, int(np.ceil(max_family_fraction * max_results)))
        keep_idx, deferred_idx, counts = _capped_rank_selection(
            order, models_arr, n_keep, cap
        )
        selected = new_template.iloc[keep_idx]
        if n_random > 0 and deferred_idx.size > 0:
            # epsilon exploration: fill remaining slots with the same log-rank
            # weighting the legacy sample uses, where index order reflects
            # children of better-scoring parents coming first. counts carries
            # over from the keep stage so the two stages share one cap budget
            # per family instead of each independently allowing up to cap.
            rest = new_template.iloc[deferred_idx].sort_index()
            fill = _capped_weighted_sample(rest, n_random, cap, counts)
            selected = pd.concat([selected, fill], axis=0)
        return selected
    except Exception as e:
        if verbose > 0:
            print(f"surrogate selection failed, using default sampling: {repr(e)}")
        return None
