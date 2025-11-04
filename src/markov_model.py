import warnings
import numpy as np
import pandas as pd

from typing import Union, Dict, List, Tuple, Optional
from pandas._libs.missing import NAType

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class MarkovModel:

    def __init__(
            self, 
            partitioner_dict: Dict[str, Tuple[int, int]],
            markov_method: str = "direct",
            rf_class_params: Optional[Dict] = None,
            rf_reg_params: Optional[Dict] = None,
            random_state: Optional[int] = 42
        ):
        """
        A Markov prediction model for forecasting fatalities.

        Args:
            partitioner_dict (Dict[str, Tuple[int, int]]): A dictionary with keys "train" and "test", 
                each mapping to a tuple of (start_month_id, end_month_id) for the respective data partitions.
            markov_method (str, optional): Forecasting method to use. Options are "direct" or "transition". 
                When "direct", the model predicts the markov state of the target month directly for any step size.
                When "transition", the model computes the transition matrix between states and uses it to forecast multiple steps ahead.
                Defaults to "direct".
            rf_class_params (Optional[Dict], optional): Parameters for Random Forest Classifier. Defaults to None.
            rf_reg_params (Optional[Dict], optional): Parameters for Random Forest Regressor. Defaults to None.
            random_state (Optional[int], optional): Random state for reproducibility. Defaults to 42.
        """

        self._train_start, self._train_end = partitioner_dict["train"]
        self._test_start, self._test_end = partitioner_dict["test"]
        self._markov_method = markov_method

        self._random_state = random_state
        self._models = {}
        self._is_fitted = False
        self._markov_states = ["peace", "desc", "esc", "war"]
        self._index_columns = ["country_id", "month_id"] #TODO should also support pgm level?
        self._target = None
        self._features = []

        # set random forest parameters
        # these are currently set to match the default parameters of the Ranger package in R,
        # where not already aligned with the default parameters of Sci-kit-learn
        # see https://cran.r-project.org/web/packages/ranger/ranger.pdf
        default_rf_class_params = {
            "n_estimators": 500,
            "random_state": self._random_state,
            "n_jobs": -1
        }
        default_rf_reg_params = {
            "n_estimators": 500,
            "max_features": "sqrt",
            "min_samples_leaf": 5,
            "random_state": self._random_state,
            "n_jobs": -1
        }

        # update with user provided parameters
        if rf_class_params is not None:
            default_rf_class_params.update(rf_class_params)
        if rf_reg_params is not None:
            default_rf_reg_params.update(rf_reg_params)

        self._rf_class_params = default_rf_class_params
        self._rf_reg_params = default_rf_reg_params


    def fit(
            self,
            data: pd.DataFrame,
            steps: int | list[int] | range,
            target: str,
            verbose: bool = True
        ) -> None:
        """
        Fit the Markov model to the provided data.
        Data must contain only the target column and feature columns, and have a multi-index with levels country_id and month_id.
        Predictions are stored in the self._models attribute.

        Args:
            data (pd.DataFrame): Input data containing features and target column.
            steps (int | list[int] | range): Steps ahead to fit the model for.
            target (str): Name of the target column in the data.
            verbose (bool, optional): Whether to print progress messages. Defaults to True.
        """

        # verify input data
        self._verify_input_data(data, target)

        # format steps to list
        steps_list = self._get_list_of_steps(steps)

        # set features
        self._features = data.columns.drop(target).tolist()

        # add markov states to data
        data = self._add_markov_states(data, target)

        # fit markov state model
        self._models["state"] = {}
        if self._markov_method == "direct":

            # if predicting directly, fit for all steps
            for step in steps_list:
                self._fit_markov_state_model(data.copy(), step, verbose)
            
        elif self._markov_method == "transition":
            # if predicting using transition matrix, only fit for step = 1
            self._fit_markov_state_model(data.copy(), step = 1, verbose = verbose)

        if verbose:
            print("\nFinished fitting Random Forest Classifiers for all Markov states.", flush=True)

        # fit fatality model
        self._fit_fatality_model(data.copy(), target, verbose)

        # set fitted flag
        self._is_fitted = True
        self._target = target


    def predict(
            self, 
            data: pd.DataFrame,
            steps: int | list[int] | range,
            verbose: bool = True
        ) -> pd.DataFrame:
        """
        Predict the target variable for the given dataset and steps.
        Data must contain the target column and feature columns, and have a multi-index with levels country_id and month_id.

        Args:
            data (pd.DataFrame): The dataset for prediction.
            steps (int | list[int] | range): Steps ahead to predict.
            verbose (bool, optional): Whether to print progress messages. Defaults to True.
        """

        if not self._is_fitted or not self._target:
            raise ValueError("Model is not yet fitted. Cannot predict")
                
        # format steps to list
        steps_list = self._get_list_of_steps(steps)

        # check if model has been trained for given steps
        self._check_if_steps_trained(steps_list)

        # add markov states to data
        data = self._add_markov_states(data, target=self._target)

        # predict for all given steps
        predictions = {}
        for step in steps_list:

            if verbose:
                print(f"Predicting step {step} using {self._markov_method} method." + " " * 20, flush=True, end="\r")

            if self._markov_method == "transition":
        
                prediction_step = self._predict_transition(
                    data.copy(),
                    step
                )
                predictions[step] = prediction_step
        
            elif self._markov_method == "direct":
            
                prediction_step = self._predict_directly(
                    data.copy(),
                    step
                )
                predictions[step] = prediction_step

        combined_predictions = pd.concat(predictions.values(), axis=0)

        # pivot to wide format
        combined_predictions = (
            combined_predictions
            .reset_index()
            .pivot(index=["country_id", "target_month_id"], columns="step", values="predicted_fatalities")
            .rename(columns=lambda x: f"predicted_fatalities_t_min_{x}")
        )

        if verbose:
            print("\nFinished predicting for all steps.", flush=True)

        return combined_predictions
    

    def _fit_markov_state_model(
            self,
            data: pd.DataFrame,
            step: int,
            verbose: bool = True
        ):
        """
        Fit the state-prediction model for a given step.
        Stores the fitted models in self._models["state"][step].

        Args:
            data (pd.DataFrame): The dataset.
            step (int): The prediction step.
            verbose (bool, optional): Whether to print progress messages. Defaults to True.
        """

        # create target state by shifting markov_state by -step
        data["markov_state_target"] = data.sort_index(level="month_id").groupby(level="country_id")["markov_state"].shift(-step)

        # add target_month_id column
        data = data.reset_index()
        data["target_month_id"] = data["month_id"] + step
        data.set_index(["country_id", "month_id"], inplace=True)

        # filter data to training period
        train_data = data.loc[
            data["target_month_id"].isin(
                range(self._train_start, self._train_end + 1))
        ].dropna().copy()

        # initialize dictionaries to store models
        rf_class_models = {}

        for state in self._markov_states:

            if verbose:
                print(f"Fitting Random Forest Classifier for state: {state} and step: {step}" + " " * 20, flush=True, end="\r")

            # get subset of data for current state
            state_subset = train_data[train_data["markov_state"] == state].drop(columns="markov_state").dropna()

            # prepare training data
            X_train = state_subset[self._features]
            y_train = state_subset["markov_state_target"]

            # initialize random forest classifier
            rf_class = RandomForestClassifier(**self._rf_class_params)

            # fit model
            rf_class.fit(X_train, y_train)

            # store model for current state
            rf_class_models[state] = rf_class

        # store all models for current step
        self._models["state"][step] = rf_class_models


    def _fit_fatality_model(
            self,
            data: pd.DataFrame,
            target: str,
            verbose: bool = True
        ):
        """
        Fit the fatality-prediction model.
        Stores the fitted models in self._models["fatalities"].

        Args:
            data (pd.DataFrame): The dataset.
            target (str): The target column name. This is used to compute markov states
            verbose (bool, optional): Whether to print progress messages. Defaults to True.
        """

        # add target column
        data["fatalities_target_month"] = data.sort_index(level="month_id").groupby(level="country_id")[target].shift(-1)
   
        # add target_month_id column
        data = data.reset_index()
        data["target_month_id"] = data["month_id"] + 1
        data.set_index(["country_id", "month_id"], inplace=True)

        # filter data to training period
        train_data = data.loc[
            data["target_month_id"].isin(
                range(self._train_start, self._train_end + 1))
        ].drop(columns="target_month_id").dropna().copy()

        rf_reg_models = {}

        for state in ["esc", "war"]:

            if verbose:
                print(f"Fitting Random Forest Regressor for state: {state}" + " " * 20, flush=True, end="\r")

            state_subset = train_data[train_data["markov_state"] == state].dropna()

            X_train = state_subset[self._features]
            y_train = state_subset["fatalities_target_month"]

            rf_reg = RandomForestRegressor(**self._rf_reg_params)

            rf_reg.fit(X_train, y_train)

            rf_reg_models[state] = rf_reg

        self._models["fatalities"] = rf_reg_models

        if verbose:
            print("\nFinished fitting Random Forest Regressors for all Markov states.", flush=True)


    def _predict_directly(
            self,
            data: pd.DataFrame,
            step: int,
        ) -> pd.DataFrame:
        """
        Predict the target variable for a given test dataset and step, using the direct method.

        Args:
            data (pd.DataFrame): The dataset.
            step (int): The prediction step.

        Returns:
            pd.DataFrame: The predicted values.
        """

        # add target_month_id column
        data = data.reset_index()
        data["target_month_id"] = data.groupby("country_id")["month_id"].shift(-step)
        data.set_index(["country_id", "month_id"], inplace=True)

          # filter data to test period
        test_data = data.loc[
            data["target_month_id"].isin(
                range(self._test_start, self._test_end + 1))
        ].copy().dropna()

        # drop non-feature columns
        X_test = test_data[self._features]

        # retrieve models for given step
        state_models = self._models["state"][step]
        fatalities_models = self._models["fatalities"]

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
        
        # ensure all required probability columns exist, if they don't, set to 0
        # this is needed because not all states will have probabilities computed
        # mainly a problem when predicting only one step ahead
        for state in self._markov_states:
            for starting_state in self._markov_states:
                if f"p_{state}_c_{starting_state}" not in test_data_full.columns:
                    test_data_full.loc[:, f"p_{state}_c_{starting_state}"] = 0


        # compute weighted fatalities
        # TODO currently this is a row-wise operation, slightly slow, can be optimized later if needed
        test_data_full["predicted_fatalities"] = test_data_full.apply(self._get_weighted_fatalities, axis=1)
        # add step as column
        test_data_full["step"] = step

        # drop rows where target_month_id is NA (due to shifting)
        test_data_full = test_data_full.dropna(subset=["target_month_id"])

        # return results
        test_data_full = (
            test_data_full
            .reset_index()
            .set_index(["country_id", "target_month_id"])
            [["predicted_fatalities", "step"]]
        )

        return test_data_full
    

    def _predict_transition(
            self,
            data: pd.DataFrame,
            step: int,
        ) -> pd.DataFrame:
        """
        Predict the target variable for a given test dataset and step using the transition method.

        Args:
            data (pd.DataFrame): The dataset.
            step (int): The prediction step. 
        Returns:
            pd.DataFrame: The predicted values.
        """

        # add target_month_id column
        data = data.reset_index()
        data["target_month_id"] = data.groupby("country_id")["month_id"].shift(-step)
        data.set_index(["country_id", "month_id"], inplace=True)

        # filter data to test period
        test_data = data.loc[
            data["target_month_id"].isin(
                range(self._test_start, self._test_end + 1))
        ].copy().dropna()

        # drop non-feature columns
        X_test = test_data[self._features]

        # retrieve models for given step
        state_models = self._models["state"][1]
        fatalities_models = self._models["fatalities"]

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

        # ensure all required probability columns exist, if they don't, set to 0
        # this is needed because not all states will have probabilities computed
        # mainly a problem when predicting only one step ahead
        for state in self._markov_states:
            for starting_state in self._markov_states:
                if f"p_{state}_c_{starting_state}" not in test_data_full.columns:
                    # print(f"Adding missing column p_{state}_c_{starting_state} with 0s")
                    test_data_full.loc[:, f"p_{state}_c_{starting_state}"] = 0

        # extract all 16 columns for the transition matrix and reshape
        n_states = len(self._markov_states)
        cols = [f"p_{next}_c_{current}" for current in self._markov_states for next in self._markov_states]

        # reshape to 3D array: (n_samples, n_states, n_states) 
        P = test_data_full[cols].to_numpy().reshape(-1, n_states, n_states)

        # compute transition matrices to the power of step
        P_k = self._matrix_power(P, step)

        # reshape back to columns of dataframe
        df_Pk = pd.DataFrame(
            P_k.reshape(-1, n_states * n_states),
            columns=cols,
            index=test_data_full.index
        )

        # merge with test data
        test_data_full.drop(columns=cols, inplace=True)
        test_data_full = test_data_full.merge(
            df_Pk,
            left_index=True,
            right_index=True,
        )

        # compute weighted fatalities
        # TODO currently this is a row-wise operation, slightly slow, can be optimized later if needed
        test_data_full["predicted_fatalities"] = test_data_full.apply(self._get_weighted_fatalities, axis=1)
        test_data_full["step"] = step

        # drop rows where target_month_id is NA (due to shifting)
        test_data_full = test_data_full.dropna(subset=["target_month_id"])

        # return results
        test_data_full = (
            test_data_full
            .reset_index()
            .set_index(["country_id", "target_month_id"])
            [["predicted_fatalities", "step"]]
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

        data = data.sort_index(level=["country_id", "month_id"])  # sort by country_id, month_id

        # compute temporary t-1 of target
        data[f"{target}_t_min_1"] = data.groupby(level="country_id")[target].shift(1)

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
    

    def _verify_input_data(
            self,
            data: pd.DataFrame,
            target: str
        ):
        """
        Verify that the data contains the required index levels and target column.

        Args:
            data (pd.DataFrame): Input data.
            target (str): Name of the target column.
        Raises:
            ValueError: If the data index does not contain required levels or if the target column is missing.
        """

        # verify index contains required levels
        if not all(col in data.index.names for col in self._index_columns):
            raise ValueError(f"Data index must contain the following levels: {self._index_columns}. Current index levels are: {data.index.names}")

        # verify target column exists
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in data columns.")
        

    def _check_if_steps_trained(
            self,
            steps: List[int]
        ) -> None:
        """
        Check if the model has been trained for the given steps.

        Args:
            steps (List[int]): Steps to check.
        Raises:
            ValueError: If the model has not been trained for any of the given steps.
        """

        trained_steps = self._models["state"].keys()

        if self._markov_method == "direct":
            for step in steps:
                if step not in trained_steps:
                    raise ValueError(f"Model has not been trained for step {step}. Please fit the model for this step before predicting.")
                
        if self._markov_method == "transition":
            if 1 not in trained_steps:
                raise ValueError("Model has not been trained for step 1 required for transition method. Please fit the model before predicting.")

    
    def _get_weighted_fatalities(self, row: pd.Series) -> float:
        """
        Compute the weighted fatalities for a given row based on Markov state probabilities.
        
        Args:
            row (pd.Series): A row of the dataframe containing probabilities and predicted fatalities.
        Returns:
            float: The weighted fatalities.
        """

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
    def _matrix_power(transition_matrix: np.ndarray, K: int) -> np.ndarray:
        """
        Compute the K-th power of a batch of transition matrices.
        Args:
            transition_matrix (np.ndarray): A 3D array of shape (n_samples, n_states, n_states) representing the transition matrices.
            K (int): The power to which to raise the matrices.
        Returns:
            np.ndarray: A 3D array of the same shape as transition_matrix, containing the K-th power of each matrix.
        """
        result = transition_matrix.copy()
        for _ in range(K - 1):
            result = np.einsum("nij,njk->nik", result, transition_matrix)
        return result

    @staticmethod
    def _get_list_of_steps(
            steps: int | list[int] | range,
        ) -> List[int]:
        """
        Formats a given steps input into a list of positive integers.

        Args:
            steps (int | list[int] | range): Steps ahead to format.
        Returns:
            List[int]: A list of positive integers representing the steps.
        Raises:
            TypeError: If steps is not an int, list of ints, or range.
            ValueError: If any step is not a positive integer.
            UserWarning: If any step is greater than 36.
        """
    
        # format steps to list
        if isinstance(steps, range):
            steps_list = list(steps)
        elif isinstance(steps, list):
            steps_list = steps
        elif isinstance(steps, int):
            steps_list = [steps]
        else:
            raise TypeError("Steps must be an int, list of ints, or range.")

        for s in steps_list:
            if not isinstance(s, int):
                raise TypeError(f"All elements in steps list must be integers. {s} is of type {type(s)}")
            
        # check that all steps are positive integers
        if any(s <= 0 for s in steps_list):
            raise ValueError("All steps must be positive integers.")
        # raise warning if steps are above 36
        if any(s > 36 for s in steps_list):
            warnings.warn("Found steps higher than 36 months. This may lead to unreliable predictions.", UserWarning)
        
        return steps_list


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