import numpy as np
import pandas as pd

from typing import Union, Dict, List, Tuple, Optional
from pandas._libs.missing import NAType

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class MarkovModel:

    def __init__(
        self, 
        partitioner_dict: Dict[str, Tuple],
        model_type: str = "both",
        random_state: Optional[int] = 42
        ):

        self._train_start, self._train_end = partitioner_dict["train"]
        self._test_start, self._test_end = partitioner_dict["test"]
        self._model_type = model_type

        self._random_state = random_state
        self._models = {}
        self._is_fitted = False
        self._markov_states = ["peace", "desc", "esc", "war"]


    def _fit_markov_state_model(
            self,
            train_data: pd.DataFrame,
            step: int,
            verbose: bool = True
        ):

        # set target markov state
        target_state_col = f"markov_state_t_plus_{step}"

        # create target column by shifting markov_state by -step
        train_data[target_state_col] = train_data.groupby(level=1)["markov_state"].shift(-step)

        rf_class_models = {}

        for state in self._markov_states:

            if verbose:
                print(f"Fitting Random Forest Classifier for state: {state}" + " " * 20, flush=True, end="\r")

            state_subset = train_data[train_data["markov_state"] == state].drop(columns="markov_state").dropna()

            X_train = state_subset.drop(columns=target_state_col)
            y_train = state_subset[target_state_col]

            rf_class = RandomForestClassifier(
                n_estimators = 200, #TODO make configurable
                random_state = self._random_state,
                class_weight = 'balanced'
            )

            rf_class.fit(X_train, y_train)

            rf_class_models[state] = rf_class

        self._models[f"rf_class_step_{step}"] = rf_class_models

        if verbose:
            print("\nFinished fitting Random Forest Classifiers for all Markov states.", flush=True)


    def _fit_fatality_model(
            self,
            data: pd.DataFrame,
            target: str,
            verbose: bool = True
        ):

        ...


    def fit(
            self,
            data: pd.DataFrame,
            step: int,
            target: str,
            verbose: bool = True
        ):

        # TODO verify that data contains correct index
        # TODO force index levels to be month_id, country_id
        # TODO verify that data contains target column, and that target is non-negative integer

        # add markov states to data
        data = self._add_markov_states(data, target)

        train_data = data.loc[
            data.index.get_level_values("month_id").isin(
                range(self._train_start, self._train_end + 1))
        ].copy()


        self._fit_markov_state_model(train_data, step, verbose)
        self._fit_fatality_model(train_data, target, verbose)

        self._is_fitted = True

        ...


    def predict(
            self, 
            steps: int,
            predict_type: str,
            EndOfHistory: Optional[int] = None
        ):

        if not self._is_fitted:
            raise ValueError("Model is not yet fitted. Cannot predict")
        
        # predict_type should be one of "train", "calibration", "future"
        if predict_type not in ["train", "calibration", "future"]:
            ...

        if predict_type == "future" and EndOfHistory is None:
            raise ValueError("EndOfHistory must be provided for future predictions")

        # should check if steps are in trained model


        ...


    
    def _add_markov_states(
            self,
            data: pd.DataFrame,
            target: str,
            threshold: int = 0
        ) -> pd.DataFrame:
        """
        Add Markov states to the data based on the target fatalities.

        Args:
            data (pd.DataFrame): Input data containing target column
            target (str): Name of target_column
            threshold (int, optional): Threshold for computing states. Defaults to 0.

        Returns:
            pd.DataFrame: Data with an additional 'markov_state' column.
        """

        data = data.sort_index(level=[1, 0])  # sort by country_id, month_id

        # compute temporary t-1 of target
        data[f"{target}_t_min_1"] = data.groupby(level=1)[target].shift(1)

        # compute markov states
        data["markov_state"] = data.apply(
            lambda row: self._compute_markov_state(
                row[target], 
                row[f"{target}_t_min_1"], 
                threshold
            ), 
            axis=1
        )

        # drop temporary t-1 column
        data.drop(columns=[f"{target}_t_min_1"], inplace=True)

        return data


    @staticmethod
    def _compute_markov_state(
            target_t: int, 
            target_t_min_1: int, 
            threshold: int = 0
        ) -> Union[str, NAType]:
        """
        Compute the Markov state based on the number of target at time t and t-1.
        Possible Markov states are:
        - "peace": target_t <= threshold and target_t_min_1 <= threshold
        - "desc": target_t <= threshold and target_t_min_1 > threshold
        - "esc": target_t > threshold and target_t_min_1 <= threshold
        - "war": target_t > threshold and target_t_min_1 > threshold
        
        Args:
            target_t (int): Target at time t.
            target_t_min_1 (int): Target at time t-1.
            threshold (int, optional): Threshold for considering target. Defaults to 0.

        Returns:
            Union[str, NAType]: The Markov state as a string or pd.NA if not computable.
        """

        if target_t <= threshold:
            if target_t_min_1 <= threshold:
                return "peace"
            elif target_t_min_1 > threshold:
                return "desc"
        elif target_t > threshold:
            if target_t_min_1 <= threshold:
                return "esc"
            elif target_t_min_1 > threshold:
                return "war"
            
        # else
        return pd.NA