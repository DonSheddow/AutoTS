# -*- coding: utf-8 -*-
"""Tests for the genetic search operators: mutation, annealing, surrogate."""
import unittest
import json
import random

import numpy as np
import pandas as pd

from autots import AutoTS
from autots.datasets import load_artificial
from autots.evaluator.auto_model import ModelMonster, NewGeneticTemplate
from autots.tools.transform import RandomTransform
from autots.evaluator.genetic_operators import (
    GENETIC_PARAMS_DEFAULTS,
    resolve_genetic_params,
    mutate_numeric_value,
    perturb_params,
    compute_generation_progress,
    annealed_parent_weights,
    featurize_template_rows,
    surrogate_select_candidates,
    numeric_bounds_from_samples,
    _capped_rank_selection,
)

TEMPLATE_COLS = ['Model', 'ModelParameters', 'TransformationParameters', 'Ensemble']


def fake_model_results(n_per_model=8, models=None):
    """Build a plausible model_results frame without running any models."""
    if models is None:
        # covers all three NewGeneticTemplate branches:
        # no_params, recombination_approved, and the else branch
        models = [
            'LastValueNaive',
            'ETS',
            'SeasonalNaive',
            'ConstantNaive',
            'AverageValueNaive',
        ]
    rows = []
    i = 0
    for model in models:
        for _ in range(n_per_model):
            rows.append(
                {
                    'ID': f"id{i}",
                    'Model': model,
                    'ModelParameters': json.dumps(
                        ModelMonster(model).get_new_params()
                    ),
                    'TransformationParameters': json.dumps(
                        RandomTransform(
                            transformer_list='fast', transformer_max_depth=2
                        )
                    ),
                    'Ensemble': 0,
                    'Exceptions': np.nan,
                    'Runs': 1,
                    'Generation': 0,
                    'ValidationRound': 0,
                    'smape': 10.0 + i * 0.7,
                    'mae': 1.0 + i * 0.13,
                    'spl': 0.5 + i * 0.11,
                    'TotalRuntimeSeconds': 1.0 + (i % 7),
                    'Score': 5.0 + i * 0.5,
                }
            )
            i += 1
    return pd.DataFrame(rows).drop_duplicates(subset=TEMPLATE_COLS)


class GeneticOperatorsTest(unittest.TestCase):
    def setUp(self):
        random.seed(2022)
        np.random.seed(2022)

    def assert_same_structure(self, original, mutated, path="root"):
        """Recursively assert perturb_params' structural invariants."""
        if isinstance(original, bool) or original is None or isinstance(original, str):
            self.assertEqual(original, mutated, path)
        elif isinstance(original, dict):
            self.assertIsInstance(mutated, dict, path)
            self.assertEqual(set(original.keys()), set(mutated.keys()), path)
            for key in original:
                self.assert_same_structure(original[key], mutated[key], f"{path}.{key}")
        elif isinstance(original, (list, tuple)):
            self.assertEqual(original, mutated, path)
        elif isinstance(original, int):
            self.assertIsInstance(mutated, int, path)
            self.assertNotIsInstance(mutated, bool, path)
            if original >= 0:
                self.assertGreaterEqual(mutated, 0, path)
        elif isinstance(original, float):
            self.assertIsInstance(mutated, float, path)
            if 0.0 < original <= 1.0:
                self.assertGreater(mutated, 0.0, path)
                self.assertLessEqual(mutated, 1.0, path)
        else:
            # numpy scalars and other exotics must pass through untouched
            self.assertEqual(original, mutated, path)

    def test_resolve_genetic_params(self):
        resolved = resolve_genetic_params(None)
        self.assertEqual(resolved, GENETIC_PARAMS_DEFAULTS)
        self.assertIsNot(resolved, GENETIC_PARAMS_DEFAULTS)
        merged = resolve_genetic_params({'surrogate': True})
        self.assertTrue(merged['surrogate'])
        self.assertTrue(merged['mutation'])
        # idempotent on an already-resolved dict
        self.assertEqual(resolve_genetic_params(merged), merged)
        with self.assertRaises(ValueError):
            resolve_genetic_params({'surogate': True})

    def test_perturb_preserves_structure_and_types(self):
        for model in ['ETS', 'ARIMA', 'GLM', 'MultivariateRegression']:
            for _ in range(20):
                params = ModelMonster(model).get_new_params()
                before = json.dumps(params, sort_keys=True, default=str)
                mutated = perturb_params(params, probability=1.0)
                # input must never be modified
                self.assertEqual(
                    before, json.dumps(params, sort_keys=True, default=str), model
                )
                self.assert_same_structure(params, mutated, path=model)

    def test_perturb_transformation_params(self):
        for _ in range(20):
            params = RandomTransform(transformer_list='fast', transformer_max_depth=4)
            for candidate in (params, json.loads(json.dumps(params))):
                mutated = perturb_params(candidate, probability=1.0)
                self.assert_same_structure(candidate, mutated)
                self.assertEqual(candidate.get('fillna'), mutated.get('fillna'))
                self.assertEqual(
                    candidate.get('transformations'), mutated.get('transformations')
                )

    def test_perturb_probability_zero_identity(self):
        for _ in range(10):
            params = ModelMonster('ARIMA').get_new_params()
            mutated = perturb_params(params, probability=0.0)
            self.assertEqual(params, mutated)

    def test_perturb_changes_something(self):
        changed = 0
        for _ in range(20):
            params = ModelMonster('ETS').get_new_params()
            mutated = perturb_params(params, probability=1.0)
            if mutated != params:
                changed += 1
        self.assertGreater(changed, 0)

    def test_perturb_handcrafted_edge_cases(self):
        params = {
            'damped_trend': True,
            'trend': None,
            'family': 'Gaussian',
            'phi': 0.999,
            'p': 1,
            'alpha': 0.5,
            'big_window': 28,
            'zero_int': 0,
            'zero_float': 0.0,
            'spacing': 5040,
            'mean_rolling_periods': [2, 4, [52, 2]],
            'regression_model': {
                'model': 'ElasticNet',
                'model_params': {'l1_ratio': 0.5, 'fit_intercept': True},
            },
            'empty': {},
            'deep': {'a': {'b': {'c': {'d': {'e': {'f': {'g': {'h': 5}}}}}}}},
        }
        for _ in range(20):
            mutated = perturb_params(params, probability=1.0)
            self.assert_same_structure(params, mutated)
            self.assertNotEqual(mutated['p'], 1)
            self.assertIn(mutated['p'], [0, 2])
            self.assertEqual(mutated['zero_int'], 1)  # -1 reflects at zero
            self.assertEqual(mutated['zero_float'], 0.0)
            self.assertNotEqual(mutated['big_window'], 28)
            self.assertEqual(mutated['regression_model']['model'], 'ElasticNet')
            # beyond max_depth the leaf is left alone
            self.assertEqual(
                mutated['deep']['a']['b']['c']['d']['e']['f']['g']['h'], 5
            )

    def test_mutate_numeric_value_types(self):
        for _ in range(50):
            self.assertIsInstance(mutate_numeric_value(12), int)
            self.assertIsInstance(mutate_numeric_value(0.4), float)
            self.assertNotEqual(mutate_numeric_value(12), 12)
            self.assertGreaterEqual(mutate_numeric_value(1), 0)

    def test_mutate_numeric_value_clamps(self):
        for _ in range(200):
            self.assertLessEqual(mutate_numeric_value(200, maximum=500), 500)
            self.assertGreaterEqual(mutate_numeric_value(200, minimum=50), 50)
            v = mutate_numeric_value(200, minimum=50, maximum=500)
            self.assertTrue(50 <= v <= 500)
            f = mutate_numeric_value(0.4, minimum=0.1, maximum=0.9)
            self.assertTrue(0.1 <= f <= 0.9)

    def test_numeric_bounds_from_samples(self):
        # a sampler with a nested cost-driver and a conditional string sibling
        def sampler():
            return {
                'model': 'RandomForest',
                'model_params': {
                    'n_estimators': random.choice([50, 100, 250, 500]),
                    'criterion': random.choice(['gini', 'entropy']),
                },
                'window': random.choice([7, 14, 28]),
            }

        bounds = numeric_bounds_from_samples(sampler, n_samples=40)
        self.assertEqual(bounds['model_params.n_estimators'], (50, 500))
        self.assertEqual(bounds['window'], (7, 28))
        # strings are not numeric leaves; 'model'/'criterion' get no bounds
        self.assertNotIn('model', bounds)
        self.assertNotIn('model_params.criterion', bounds)

    def test_bounded_mutation_confines_cost_driver(self):
        # a wide-range integer cost-driver must never drift outside the
        # sampled range no matter how many generations a lineage survives
        bounds = {'model_params.n_estimators': (50, 500)}
        params = {'model': 'RandomForest', 'model_params': {'n_estimators': 100}}
        maxed = 0
        for _ in range(200):  # far more than any real lineage
            params = perturb_params(params, probability=1.0, bounds=bounds)
            n = params['model_params']['n_estimators']
            self.assertTrue(50 <= n <= 500, f"n_estimators drifted to {n}")
            maxed = max(maxed, n)
        # sanity: it actually moves around within the band, not frozen
        self.assertGreater(maxed, 100)

    def test_perturb_bounds_only_touch_sanctioned_paths(self):
        # a numeric leaf whose path was never sampled is left untouched
        params = {'covered': 100, 'uncovered': 100}
        bounds = {'covered': (50, 500)}
        for _ in range(50):
            out = perturb_params(params, probability=1.0, bounds=bounds)
            self.assertEqual(out['uncovered'], 100)

    def test_compute_generation_progress(self):
        self.assertAlmostEqual(compute_generation_progress(5, 25, 1, 9e6), 0.2)
        self.assertAlmostEqual(compute_generation_progress(3, 99999, 30, 60), 0.5)
        self.assertAlmostEqual(compute_generation_progress(3, 99999, 30, 9e6), 0.5)
        self.assertAlmostEqual(compute_generation_progress(30, 25, 1, 9e6), 1.0)

    def test_annealed_parent_weights(self):
        self.assertIsNone(annealed_parent_weights('model_recombination', None))
        lengths = {
            'model_recombination': 4,
            'transformer': 5,
            'keep_fresh_mutate': 3,
            'keep_fresh': 2,
        }
        for kind, length in lengths.items():
            for progress in [0.0, 0.5, 1.0]:
                weights = annealed_parent_weights(kind, progress)
                self.assertEqual(len(weights), length)
                self.assertAlmostEqual(sum(weights), 1.0)
        early = annealed_parent_weights('model_recombination', 0.0)
        late = annealed_parent_weights('model_recombination', 1.0)
        # fresh randoms (index 2) favored early, elites (index 0) late
        self.assertGreater(early[2], early[0])
        self.assertGreater(late[0], late[2])

    def test_featurize_template_rows(self):
        template = pd.DataFrame(
            {
                'Model': ['ETS', 'ETS', 'ARIMA', 'Broken'],
                'ModelParameters': [
                    '{"trend": null, "seasonal_periods": 7}',
                    '{"trend": null, "seasonal_periods": 7}',
                    '{"p": 2, "d": 1, "q": 0}',
                    'not valid json',
                ],
                'TransformationParameters': ['{}', '{}', '{}', '{}'],
            }
        )
        X = featurize_template_rows(template, n_features=512)
        self.assertEqual(X.shape, (4, 512))
        self.assertTrue(np.array_equal(X[0], X[1]))
        self.assertFalse(np.array_equal(X[0], X[2]))
        X2 = featurize_template_rows(template, n_features=512)
        self.assertTrue(np.array_equal(X, X2))

    def test_capped_rank_selection(self):
        order = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        models = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'C', 'C'])

        selected, deferred, counts = _capped_rank_selection(order, models, k=4, cap=2)
        self.assertEqual(len(selected), 4)
        self.assertEqual(set(selected.tolist()) | set(deferred.tolist()), set(order.tolist()))
        actual_counts = pd.Series(models[selected]).value_counts()
        self.assertTrue((actual_counts <= 2).all())
        self.assertEqual(counts, actual_counts.to_dict())
        # best-first order still respected within the cap: A and B (the
        # earliest-ranked families) win out over C
        self.assertEqual(set(models[selected]), {'A', 'B'})

        # too few distinct families to fill k under a tight cap -> backfill
        # still reaches k rather than returning a short list
        selected2, deferred2, counts2 = _capped_rank_selection(order, models, k=4, cap=1)
        self.assertEqual(len(selected2), 4)
        self.assertEqual(set(selected2.tolist()) | set(deferred2.tolist()), set(order.tolist()))
        self.assertEqual(len(set(models[selected2])), 3)  # cap forced all 3 families in

        # k larger than the pool returns everything without error
        selected3, deferred3, counts3 = _capped_rank_selection(order, models, k=100, cap=10)
        self.assertEqual(len(selected3), len(order))
        self.assertEqual(len(deferred3), 0)

        # counts continues from a caller-supplied running tally (used to
        # share one cap budget between the surrogate's two selection stages)
        selected4, _, counts4 = _capped_rank_selection(
            order, models, k=2, cap=2, counts={'A': 2}
        )
        self.assertNotIn('A', models[selected4])  # A already at cap, skipped
        self.assertEqual(counts4['A'], 2)

    def test_surrogate_family_cap(self):
        random.seed(7)
        np.random.seed(7)

        def rows(model, knob_range, n):
            knobs = np.random.uniform(*knob_range, size=n)
            return pd.DataFrame(
                {
                    'Model': model,
                    'ModelParameters': [json.dumps({'knob': float(k)}) for k in knobs],
                    'TransformationParameters': '{}',
                    'Ensemble': 0,
                    'Exceptions': np.nan,
                    'Score': knobs,  # lower knob -> lower (better) score
                }
            )

        # 'Bland' is a cheap family scored well regardless of its specific
        # params, historically -- the pattern that made the uncapped
        # surrogate collapse onto naive models in the real search.
        history = pd.concat(
            [
                rows('Bland', (0, 5), 80),
                rows('Sharp', (10, 40), 40),
                rows('Solid', (10, 40), 40),
                rows('Rare', (10, 40), 40),
            ],
            ignore_index=True,
        )
        candidates = pd.concat(
            [
                rows('Bland', (0, 5), 40),
                rows('Sharp', (10, 40), 20),
                rows('Solid', (10, 40), 20),
                rows('Rare', (10, 40), 20),
            ],
            ignore_index=True,
        )[['Model', 'ModelParameters', 'TransformationParameters', 'Ensemble']]

        uncapped = surrogate_select_candidates(
            candidates, history, max_results=20, keep_fraction=0.8,
            max_family_fraction=1.0,
        )
        capped = surrogate_select_candidates(
            candidates, history, max_results=20, keep_fraction=0.8,
            max_family_fraction=0.35,
        )
        self.assertEqual(uncapped.shape[0], 20)
        self.assertEqual(capped.shape[0], 20)
        uncapped_bland = (uncapped['Model'] == 'Bland').sum()
        capped_bland = (capped['Model'] == 'Bland').sum()
        # uncapped collapses heavily onto the historically-safe family...
        self.assertGreater(uncapped_bland, 10)
        # ...capped keeps it dominant but bounded, and other families survive
        self.assertLessEqual(capped_bland, 7)  # ceil(0.35 * 20)
        self.assertGreaterEqual(capped['Model'].nunique(), 3)

    def test_surrogate_select_candidates(self):
        def frame(knobs, score_offset=0.0):
            return pd.DataFrame(
                {
                    'Model': 'FakeModel',
                    'ModelParameters': [json.dumps({'knob': k}) for k in knobs],
                    'TransformationParameters': '{}',
                    'Ensemble': 0,
                    'Exceptions': np.nan,
                    'Score': [k + score_offset for k in knobs],
                }
            )

        history = frame([float(i) for i in range(120)])
        failures = frame([200.0 + i for i in range(5)])
        failures['Exceptions'] = 'ValueError()'
        failures['Score'] = np.nan
        history = pd.concat([history, failures], ignore_index=True)
        candidates = frame([float(i) for i in range(60)])[
            ['Model', 'ModelParameters', 'TransformationParameters', 'Ensemble']
        ]
        selected = surrogate_select_candidates(
            candidates, history, max_results=20, keep_fraction=0.8
        )
        self.assertEqual(selected.shape[0], 20)
        selected_knobs = [
            json.loads(x)['knob'] for x in selected['ModelParameters']
        ]
        pool_mean = np.mean(range(60))
        self.assertLess(np.mean(selected_knobs), pool_mean * 0.8)
        # not enough history returns None -> caller falls back to legacy path
        self.assertIsNone(
            surrogate_select_candidates(
                candidates, history.head(10), max_results=20
            )
        )

    def test_new_genetic_template_paths(self):
        results = fake_model_results(n_per_model=12)
        submitted = results[TEMPLATE_COLS]
        model_list = results['Model'].unique().tolist()
        common_args = dict(
            submitted_parameters=submitted,
            sort_column="Score",
            sort_ascending=True,
            max_results=20,
            max_per_model_class=5,
            top_n=25,
            template_cols=TEMPLATE_COLS,
            transformer_list='fast',
            transformer_max_depth=2,
            model_list=model_list,
        )
        legacy = NewGeneticTemplate(
            results,
            genetic_params={'mutation': False, 'anneal': False, 'surrogate': False},
            **common_args,
        )
        with_new = NewGeneticTemplate(
            results, genetic_params=None, generation_progress=0.9, **common_args
        )
        with_surrogate = NewGeneticTemplate(
            results,
            genetic_params={'surrogate': True, 'surrogate_min_rows': 50},
            generation_progress=0.5,
            **common_args,
        )
        for name, template in [
            ('legacy', legacy),
            ('with_new', with_new),
            ('with_surrogate', with_surrogate),
        ]:
            self.assertFalse(template.empty, name)
            self.assertLessEqual(template.shape[0], 20, name)
            self.assertEqual(list(template.columns), TEMPLATE_COLS, name)
            self.assertTrue((template['Ensemble'] == 0).all(), name)
            for col in ['ModelParameters', 'TransformationParameters']:
                for param_json in template[col]:
                    json.loads(param_json)
            overlap = template.merge(submitted, on=TEMPLATE_COLS, how='inner')
            self.assertTrue(overlap.empty, name)
        self.assertEqual(with_surrogate.shape[0], 20)

        # same-seed determinism of the new default path
        random.seed(123)
        np.random.seed(123)
        first = NewGeneticTemplate(
            results, genetic_params=None, generation_progress=0.9, **common_args
        )
        random.seed(123)
        np.random.seed(123)
        second = NewGeneticTemplate(
            results, genetic_params=None, generation_progress=0.9, **common_args
        )
        pd.testing.assert_frame_equal(
            first.reset_index(drop=True), second.reset_index(drop=True)
        )


class GeneticE2ETest(unittest.TestCase):
    def test_autots_genetic_params_e2e(self):
        df = load_artificial(long=False).iloc[:, :4]
        model = AutoTS(
            forecast_length=4,
            frequency='infer',
            max_generations=3,
            model_list=[
                'ConstantNaive',
                'LastValueNaive',
                'AverageValueNaive',
                'SeasonalNaive',
            ],
            initial_template='Random',
            num_validations=1,
            genetic_params={'surrogate': True, 'surrogate_min_rows': 20},
            ensemble=None,
            verbose=-1,
            n_jobs=1,
        )
        model = model.fit(df)
        prediction = model.predict(verbose=-1)
        self.assertEqual(prediction.forecast.shape, (4, df.shape[1]))
        results = model.results()
        self.assertGreater(results['Generation'].max(), 1)
        self.assertTrue(model.genetic_params['surrogate'])


if __name__ == '__main__':
    unittest.main()
