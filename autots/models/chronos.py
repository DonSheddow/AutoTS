"""Chronos-2 pretrained foundation model wrapper.

Wraps Amazon's Chronos-2 (`chronos-forecasting` package) as an AutoTS model.
Chronos-2 is a zero-shot forecaster that natively produces quantile forecasts
and supports covariate-informed forecasting, so "fitting" simply retains the
training context the model conditions on at predict time.
"""

import random
import datetime
import numpy as np
import pandas as pd
from autots.models.base import ModelObject, PredictionObject


class Chronos2(ModelObject):
    """Chronos-2 pretrained zero-shot forecasting model.

    Args:
        name (str): Model name.
        frequency (str): String alias of datetime index frequency or else 'infer'.
        prediction_interval (float): Confidence interval for probabilistic forecast.
        model_name (str): HuggingFace checkpoint id, e.g. "amazon/chronos-2".
        context_length (int): Max history length passed to Chronos-2.
        device_map (str): torch device map. None auto-picks 'cuda' if available
            else 'cpu'. Pass through e.g. 'cuda', 'cpu', 'mps'.
        torch_dtype (str): dtype for the model, e.g. 'auto', 'bfloat16', 'float32'.
        cross_learning (bool): If True, share patterns across all series.
        multivariate (bool): If True, treat all columns as one joint group
            (currently mapped onto cross_learning).
        batch_size (int): Batch size for inference.
        regression_type (str): "User" enables covariates from future_regressor.
        forecast_length (int): Accepted for interface parity; the actual horizon
            comes from predict().
    """

    def __init__(
        self,
        name: str = "Chronos2",
        frequency: str = "infer",
        prediction_interval: float = 0.9,
        model_name: str = "amazon/chronos-2",
        context_length: int = 2048,
        device_map: str = None,
        torch_dtype: str = "auto",
        cross_learning: bool = False,
        multivariate: bool = False,
        batch_size: int = 256,
        regression_type: str = None,
        forecast_length: int = 14,
        holiday_country: str = "US",
        random_seed: int = 2020,
        verbose: int = 0,
        n_jobs: int = "auto",
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name=name,
            frequency=frequency,
            prediction_interval=prediction_interval,
            regression_type=regression_type,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.model_name = model_name
        self.context_length = context_length
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.cross_learning = cross_learning
        self.multivariate = multivariate
        self.batch_size = batch_size
        self.forecast_length = forecast_length

    def _resolve_device(self):
        if self.device_map is not None:
            return self.device_map
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def fit(self, df, future_regressor=None):
        """Load the pretrained pipeline and retain training context.

        Args:
            df (pandas.DataFrame): Datetime Indexed, wide style data (one column
                per series).
            future_regressor (pandas.DataFrame): Optional wide regressors aligned
                to df.index, used when regression_type == "User".
        """
        df = self.basic_profile(df)
        try:
            from chronos import Chronos2Pipeline
        except ImportError:
            raise ImportError(
                "Chronos2 requires the chronos-forecasting package. "
                "Install it with `pip install chronos-forecasting`."
            )

        self.pipeline = Chronos2Pipeline.from_pretrained(
            self.model_name,
            device_map=self._resolve_device(),
            dtype=self.torch_dtype,
        )
        self.df_train = df

        self.regressor_train = None
        if self.regression_type in ["User", "user"] and future_regressor is not None:
            self.regressor_train = pd.DataFrame(future_regressor).reindex(df.index)

        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def _build_long(self, wide, regressor=None):
        """Convert a wide df (and optional shared regressor) to Chronos-2 long
        format with one item_id per series."""
        long_df = wide.reset_index().melt(
            id_vars=[wide.index.name or "index"],
            var_name="item_id",
            value_name="target",
        )
        long_df = long_df.rename(columns={wide.index.name or "index": "timestamp"})
        if regressor is not None and not regressor.empty:
            # broadcast the shared regressor onto every series via timestamp merge
            reg = regressor.copy()
            reg.index.name = "timestamp"
            reg = reg.reset_index()
            long_df = long_df.merge(reg, on="timestamp", how="left")
        return long_df

    def predict(
        self,
        forecast_length="self",
        future_regressor=None,
        just_point_forecast=False,
    ):
        """Generate forecasts.

        Args:
            forecast_length (int): Number of periods to forecast.
            future_regressor (pandas.DataFrame): Known-future regressors for the
                horizon, used when regression_type == "User".
            just_point_forecast (bool): If True return only the point forecast df.
        """
        predictStartTime = datetime.datetime.now()
        if forecast_length == "self":
            forecast_length = self.forecast_length

        lower_q = round((1 - self.prediction_interval) / 2, 4)
        upper_q = round(1 - lower_q, 4)
        quantile_levels = [lower_q, 0.5, upper_q]

        use_covariates = (
            self.regression_type in ["User", "user"]
            and self.regressor_train is not None
            and future_regressor is not None
        )

        forecast_index = self.create_forecast_index(forecast_length)

        context_long = self._build_long(
            self.df_train,
            regressor=self.regressor_train if use_covariates else None,
        )
        future_long = None
        if use_covariates:
            future_reg = pd.DataFrame(future_regressor).iloc[:forecast_length]
            future_reg.index = forecast_index
            # future_df carries only the (known-future) covariate columns
            future_long = self._build_long(
                pd.DataFrame(index=forecast_index, columns=self.column_names),
                regressor=future_reg,
            ).drop(columns=["target"])

        pred_df = self.pipeline.predict_df(
            context_long,
            future_df=future_long,
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
            prediction_length=forecast_length,
            quantile_levels=quantile_levels,
            batch_size=self.batch_size,
            context_length=self.context_length,
            cross_learning=self.cross_learning or self.multivariate,
            freq=self.frequency,
        )

        def _pivot(value_col):
            wide = pred_df.pivot(
                index="timestamp", columns="item_id", values=value_col
            )
            wide = wide.reindex(columns=self.column_names)
            wide.index = forecast_index
            return wide

        forecast = _pivot("predictions")
        lower_forecast = _pivot(str(lower_q))
        upper_forecast = _pivot(str(upper_q))

        if just_point_forecast:
            return forecast

        predict_runtime = datetime.datetime.now() - predictStartTime
        prediction = PredictionObject(
            model_name=self.name,
            forecast_length=forecast_length,
            forecast_index=forecast.index,
            forecast_columns=self.column_names,
            lower_forecast=lower_forecast,
            forecast=forecast,
            upper_forecast=upper_forecast,
            prediction_interval=self.prediction_interval,
            predict_runtime=predict_runtime,
            fit_runtime=self.fit_runtime,
            model_parameters=self.get_params(),
        )
        return prediction

    def get_new_params(self, method: str = "random"):
        """Return dict of new parameters for parameter tuning."""
        if "regressor" in method:
            regression_choice = "User"
        else:
            regression_choice = random.choices([None, "User"], [0.7, 0.3])[0]
        return {
            "model_name": random.choices(
                [
                    "amazon/chronos-2",
                    "autogluon/chronos-2-small",
                    "autogluon/chronos-2-synth",
                ],
                [0.6, 0.25, 0.15],
            )[0],
            "context_length": random.choice([512, 1024, 2048, 4096]),
            "cross_learning": random.choices([False, True], [0.8, 0.2])[0],
            "multivariate": random.choices([False, True], [0.85, 0.15])[0],
            "batch_size": random.choice([128, 256, 512]),
            "regression_type": regression_choice,
        }

    def get_params(self):
        """Return dict of current parameters."""
        return {
            "model_name": self.model_name,
            "context_length": self.context_length,
            "cross_learning": self.cross_learning,
            "multivariate": self.multivariate,
            "batch_size": self.batch_size,
            "regression_type": self.regression_type,
        }
