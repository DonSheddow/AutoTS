# -*- coding: utf-8 -*-
"""Tests that ensemble component models survive a round trip.

Regression tests for ensembles losing their `models` dict: the component
forecasts used to be keyed by a hash of the parameters each component
*returned*, which does not match the ID the ensemble template references
whenever a component alters its own parameters during fit (every nested
Ensemble does, see BestNEnsemble). Those components were then dropped from
`models` in the recorded/exported parameters, and an ensemble made only of
ensembles ended up with `models = {}` - scoring fine, then raising
"BestN failed, no component models available." on the next predict.
"""
import unittest
import json

import numpy as np
import pandas as pd

from autots.evaluator.auto_model import (
    model_forecast,
    create_model_id,
    drop_empty_ensembles,
)
from autots.models.ensemble import BestNEnsemble


def _df(n=120, cols=("a", "b")):
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {col: np.cumsum(rng.normal(size=n)) + 100 for col in cols}, index=idx
    )


def _member(model, params=None, trans=None):
    """(id, template entry) as stored in an ensemble's `models` dict."""
    params = {} if params is None else params
    trans = {} if trans is None else trans
    entry = {
        "Model": model,
        "ModelParameters": json.dumps(params),
        "TransformationParameters": json.dumps(trans),
    }
    return create_model_id(model, params, trans), entry


def _bestn(members, metric="test", point_method=None):
    params = {
        "model_name": "BestN",
        "model_count": len(members),
        "model_metric": metric,
        "models": dict(members),
    }
    if point_method is not None:
        params["point_method"] = point_method
    return params


def _run(params, df, forecast_length=5, verbose=0):
    return model_forecast(
        model_name="Ensemble",
        model_param_dict=json.dumps(params),
        model_transform_dict="{}",
        df_train=df,
        forecast_length=forecast_length,
        frequency="D",
        verbose=verbose,
    )


class TestEnsembleComponentRetention(unittest.TestCase):
    def setUp(self):
        self.df = _df()

    def test_plain_components_retained(self):
        params = _bestn([_member("LastValueNaive"), _member("AverageValueNaive")])
        result = _run(params, self.df)
        self.assertEqual(
            set(result.model_parameters["models"].keys()), set(params["models"].keys())
        )

    def test_nested_ensemble_components_retained(self):
        """The crash case: an ensemble whose members are themselves ensembles."""
        inner1 = _bestn([_member("LastValueNaive"), _member("AverageValueNaive")], "i1")
        inner2 = _bestn([_member("SeasonalNaive"), _member("AverageValueNaive")], "i2")
        outer = _bestn(
            [_member("Ensemble", inner1), _member("Ensemble", inner2)], "outer"
        )
        result = _run(outer, self.df)
        self.assertEqual(
            set(result.model_parameters["models"].keys()), set(outer["models"].keys())
        )
        # and what was recorded must still be runnable, which is what failed
        # at the final predict of a run
        rerun = _run(result.model_parameters, self.df)
        self.assertEqual(rerun.forecast.shape, result.forecast.shape)

    def test_mixed_nested_ensemble_components_retained(self):
        inner = _bestn([_member("LastValueNaive"), _member("AverageValueNaive")], "i")
        mixed = _bestn([_member("SeasonalNaive"), _member("Ensemble", inner)], "mixed")
        result = _run(mixed, self.df)
        self.assertEqual(len(result.model_parameters["models"]), 2)

    def test_failed_component_dropped_others_kept(self):
        """A genuinely broken component is still pruned from the params."""
        good_id, good = _member("LastValueNaive")
        bad_id, bad = _member("NotAModelName")
        params = _bestn([(good_id, good), (bad_id, bad)])
        result = _run(params, self.df)
        self.assertEqual(list(result.model_parameters["models"].keys()), [good_id])

    def test_all_components_failing_raises(self):
        params = _bestn([_member("NotAModelName"), _member("AlsoNotAModel")])
        with self.assertRaises(ValueError):
            _run(params, self.df)

    def test_empty_models_raises_before_running(self):
        params = _bestn([])
        with self.assertRaises(ValueError):
            _run(params, self.df)


class TestBestNEnsembleParams(unittest.TestCase):
    """BestNEnsemble mutates the params that get recorded and exported."""

    def setUp(self):
        idx = pd.date_range("2022-01-01", periods=5, freq="D")
        self.fore = pd.DataFrame({"a": np.arange(5.0)}, index=idx)

    def _call(self, params, forecasts):
        return BestNEnsemble(
            params,
            forecasts,
            {k: v - 1 for k, v in forecasts.items()},
            {k: v + 1 for k, v in forecasts.items()},
            {k: pd.Timedelta(seconds=1) for k in forecasts},
        )

    def test_never_empties_models_on_id_mismatch(self):
        """Unknown forecast keys must not wipe the template out."""
        m1, m2 = _member("LastValueNaive"), _member("AverageValueNaive")
        params = _bestn([m1, m2])
        self._call(params, {"some-other-id": self.fore})
        self.assertEqual(set(params["models"].keys()), {m1[0], m2[0]})

    def test_prunes_to_models_that_ran(self):
        m1, m2 = _member("LastValueNaive"), _member("AverageValueNaive")
        params = _bestn([m1, m2])
        self._call(params, {m1[0]: self.fore})
        self.assertEqual(list(params["models"].keys()), [m1[0]])

    def test_empty_template_models_raises(self):
        with self.assertRaises(ValueError):
            self._call(_bestn([]), {"anything": self.fore})

    def test_no_forecasts_raises(self):
        with self.assertRaises(ValueError):
            self._call(_bestn([_member("LastValueNaive")]), {})


class TestDropEmptyEnsembles(unittest.TestCase):
    def test_drops_only_empty_ensembles(self):
        good = _bestn([_member("LastValueNaive")])
        template = pd.DataFrame(
            {
                "Model": ["Ensemble", "Ensemble", "LastValueNaive"],
                "ModelParameters": [
                    json.dumps(_bestn([])),
                    json.dumps(good),
                    "{}",
                ],
                "TransformationParameters": ["{}"] * 3,
                "Ensemble": [1, 1, 0],
            }
        )
        result = drop_empty_ensembles(template)
        self.assertEqual(result["Model"].tolist(), ["Ensemble", "LastValueNaive"])
        self.assertEqual(
            json.loads(result.iloc[0]["ModelParameters"])["models"].keys(),
            good["models"].keys(),
        )

    def test_handles_unrelated_frames(self):
        empty = pd.DataFrame()
        self.assertTrue(drop_empty_ensembles(empty).empty)
        other = pd.DataFrame({"x": [1]})
        self.assertEqual(drop_empty_ensembles(other).shape, (1, 1))


class TestTemplateGuards(unittest.TestCase):
    """Degenerate ensembles must not survive a template round trip."""

    def setUp(self):
        from autots import AutoTS

        self.model = AutoTS(
            forecast_length=5, frequency="D", model_list="superfast", verbose=-1
        )
        self.empty_row = {
            "Model": "Ensemble",
            "ModelParameters": json.dumps(_bestn([])),
            "TransformationParameters": "{}",
            "Ensemble": 1,
        }
        self.plain_row = {
            "Model": "LastValueNaive",
            "ModelParameters": "{}",
            "TransformationParameters": "{}",
            "Ensemble": 0,
        }

    def test_import_template_drops_empty_ensemble(self):
        template = pd.DataFrame([self.empty_row, self.plain_row])
        self.model.import_template(
            template, method="only", enforce_model_list=False, include_ensemble=True
        )
        self.assertEqual(
            self.model.initial_template["Model"].tolist(), ["LastValueNaive"]
        )

    def test_import_best_model_skips_empty_ensemble(self):
        template = pd.DataFrame([self.empty_row, self.plain_row])
        self.model.import_best_model(template, enforce_model_list=False)
        self.assertEqual(self.model.best_model_name, "LastValueNaive")

    def test_import_best_model_all_empty_raises(self):
        with self.assertRaises(ValueError):
            self.model.import_best_model(
                pd.DataFrame([self.empty_row]), enforce_model_list=False
            )

    def test_export_drops_empty_ensemble(self):
        exported = self.model.save_template(
            None, pd.DataFrame([self.empty_row, self.plain_row])
        )
        self.assertEqual(exported["Model"].tolist(), ["LastValueNaive"])


if __name__ == '__main__':
    unittest.main()
