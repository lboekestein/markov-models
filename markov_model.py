import numpy as np
import pandas as pd

from typing import Union, Dict, List, Tuple, Optional
from pandas._libs.missing import NAType

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class MarkovModel:

    def __init__(
            self, 
            partitioner_dict: Dict[str, Tuple[int, int]],
            model_type: str = "both",
            markov_method: str = "direct",
            random_state: Optional[int] = 42
        ):
        """
        A Markov prediction model for forecasting fatalities.

        Args:
            partitioner_dict (Dict[str, Tuple[int, int]]): A dictionary with keys "train" and "test", 
                each mapping to a tuple of (start_month_id, end_month_id) for the respective data partitions.
            model_type (str, optional): Type of model to use for the regression step. Options are: 
                - "rf" for a Random Forest Regression model;
                - "glm" for Generalized Linear Model; 
                - or "both" for both models. Defaults to "both".
            markov_method (str, optional): Forecasting method to use. Options are "direct" or "transition". 
                When "direct", the model predicts the markov state of the target month directly for any step size.
                When "transition", the model computes the transition matrix between states and uses it to forecast multiple steps ahead.
                Defaults to "direct".
            random_state (Optional[int], optional): Random state for reproducibility. Defaults to 42.
        """

        self._train_start, self._train_end = partitioner_dict["train"]
        self._test_start, self._test_end = partitioner_dict["test"]
        self._model_type = model_type
        self._markov_method = markov_method

        self._random_state = random_state
        self._models = {}
        self._is_fitted = False
        self._markov_states = ["peace", "desc", "esc", "war"]


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

        # filter data to training period
        train_data = data.loc[
            data.index.get_level_values("month_id").isin(
                range(self._train_start, self._train_end + 1))
        ].copy()

        # fit markov state model

        # TODO support multiple steps
        # TODO distinguish between markov_method options
        # TODO add options for model_type argument

        self._fit_markov_state_model(train_data.copy(), step, verbose)
        self._fit_fatality_model(train_data.copy(), step, target, verbose)

        self._is_fitted = True


    def predict(
            self, 
            data: pd.DataFrame,
            target: str,
            step: int,
            predict_type: str,
            EndOfHistory: Optional[int] = None
        ):

        if not self._is_fitted:
            raise ValueError("Model is not yet fitted. Cannot predict")
        
        # predict_type should be one of "train", "calibration", "future"
        if predict_type not in ["calibration", "future"]:
            ...

        if predict_type == "future" and EndOfHistory is None:
            raise ValueError("EndOfHistory must be provided for future predictions")

        # TODO should check if steps are in trained model
        # TODO support multiple steps
        # TODO remove target column requirement, save in fitting step

        # add markov states to data
        data = self._add_markov_states(data, target=target)

        # filter data to test period
        test_data = data.loc[
            data.index.get_level_values("month_id").isin(
                range(self._test_start, self._test_end + 1))
        ].copy()

        # add target_month_id column
        test_data = test_data.reset_index()
        test_data["target_month_id"] = test_data.groupby("country_id")["month_id"].shift(-step)
        test_data.set_index(["country_id", "month_id"], inplace=True)

        return self._predict_by_step(test_data, step)


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

        self._models["state"] = {}
        rf_class_models = {}

        for state in self._markov_states:

            if verbose:
                print(f"Fitting Random Forest Classifier for state: {state}" + " " * 20, flush=True, end="\r")

            state_subset = train_data[train_data["markov_state"] == state].drop(columns="markov_state").dropna()

            X_train = state_subset.drop(columns=target_state_col)
            y_train = state_subset[target_state_col]

            rf_class = RandomForestClassifier(
                n_estimators = 200,                #TODO make configurable
                random_state = self._random_state,
                class_weight = 'balanced',
            )

            rf_class.fit(X_train, y_train)

            rf_class_models[state] = rf_class

        self._models["state"][step] = rf_class_models

        if verbose:
            print("\nFinished fitting Random Forest Classifiers for all Markov states.", flush=True)


    def _fit_fatality_model(
            self,
            train_data: pd.DataFrame,
            step: int,
            target: str,
            verbose: bool = True
        ):

        # set target column
        target_column = f"fatalities_t_plus_{step}"

        # add target column
        train_data[target_column] = train_data.groupby(level=1)[target].shift(-step)

        self._models["fatalities"] = {}
        rf_reg_models = {}

        for state in ["esc", "war"]:

            if verbose:
                print(f"Fitting Random Forest Regressor for state: {state}" + " " * 20, flush=True, end="\r")

            state_subset = train_data[train_data["markov_state"] == state].drop(columns="markov_state").dropna()

            X_train = state_subset.drop(columns=target_column)
            y_train = state_subset[target_column]

            rf_reg = RandomForestRegressor(
                n_estimators=200,
                random_state=self._random_state,
            )

            rf_reg.fit(X_train, y_train)

            rf_reg_models[state] = rf_reg

        self._models["fatalities"][step] = rf_reg_models

        if verbose:
            print("\nFinished fitting Random Forest Regressors for all Markov states.", flush=True)


    def _predict_by_step(
            self,
            test_data: pd.DataFrame,
            step: int,
        ) -> pd.DataFrame:
        """
        Predict the target variable for a given test dataset and step.

        Args:
            test_data (pd.DataFrame): The test dataset.
            step (int): The prediction step.

        Returns:
            pd.DataFrame: The predicted values.
        """

        # drop non-feature columns
        X_test = test_data.drop(columns=["markov_state", "target_month_id"])

        # retrieve models for given step
        state_models = self._models["state"][step]
        fatalities_models = self._models["fatalities"][step]

        # Initialize lists to hold results
        state_probabilities = []
        predicted_fatalities = []

        # iterate over each possible starting state
        for start_state in self._markov_states:
            
            # 1) predict probability of markov states in target month given current state
            state_probs = state_models[start_state].predict_proba(X_test)
            
            # add to results
            state_probabilities.append(
                pd.DataFrame(state_probs, 
                            columns=[f"p_{next}_c_{start_state}" 
                                    for next in state_models[start_state].classes_], 
                            index=X_test.index))

            # 2) predict fatalities in target month given current state (only for esc and war)
            if start_state in ["esc", "war"]:

                # predict fatalities given start state
                fatalities_preds = fatalities_models[start_state].predict(X_test)

                # add to results
                predicted_fatalities.append(pd.Series(fatalities_preds, 
                                                    index=X_test.index, 
                                                    name=f"predicted_fatalities_c_{start_state}"))

        # Concatenate all start-state probability tables horizontally
        state_probabilities_df = pd.concat(state_probabilities, axis=1)
        # Concatenate all predicted fatalities tables horizontally
        predicted_fatalities_df = pd.concat(predicted_fatalities, axis=1)

        # combine results with test data
        test_data_full = pd.concat([
            test_data, state_probabilities_df, predicted_fatalities_df
            ], axis=1)

        # compute weighted fatalities
        test_data_full["predicted_fatalities"] = test_data_full.apply(self._get_weighted_fatalities, axis=1)

        # drop rows where target_month_id is NA (due to shifting)
        test_data_full = test_data_full.dropna(subset=["target_month_id"])

        # return results
        test_data_full = (
            test_data_full
            .reset_index()
            .set_index(["country_id", "target_month_id"])
            [["predicted_fatalities"]]
        )

        return test_data_full

  
    
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
    def _get_weighted_fatalities(row):

        # set current state
        current_state = row["markov_state"]
        
        # weight fatalities based on markov state probabilities
        weighted_fatalities = (
            # predicted fatalities conditional on peace and descalation are 0, but still include for clarity
            row[f"p_peace_c_{current_state}"] * 0 +
            row[f"p_desc_c_{current_state}"] * 0 +
            row[f"p_esc_c_{current_state}"] * row[f"predicted_fatalities_c_esc"] +
            row[f"p_war_c_{current_state}"] * row[f"predicted_fatalities_c_war"]
        )

        return weighted_fatalities


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