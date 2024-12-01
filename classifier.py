import os
import json
from sklearn.model_selection import StratifiedKFold, cross_validate
from tqdm import tqdm
import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Any, List, Optional, TextIO
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint, loguniform
import matplotlib.pyplot as plt

hidden_layer_sizes = [
    tuple(np.random.randint(10, 64) for _ in range(np.random.randint(1, 3)))
    for _ in range(100)  # Generate 100 random configurations
]
random_param_dist = {
    "LR": {
        'classifier__C': loguniform(1e-4, 1e2),
        'classifier__penalty': ['l2'],
        'classifier__solver': ['lbfgs', 'saga'],
        'classifier__max_iter': randint(50, 5001),
        'classifier__class_weight': ['balanced', None],
        'classifier__tol': loguniform(1e-4, 1e-1)
    },
    "SVC": {
        'classifier__C': loguniform(1e-6, 1e3),   # Regularization parameter
        'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel types to try
        'classifier__degree': randint(2, 6),       # Degree of polynomial kernel
        'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],  # Predefined gamma values
        'classifier__coef0': loguniform(1e-2, 1e0),  # Adjusted for positive values
        'classifier__shrinking': [True, False],     # Whether to use shrinking heuristic
        'classifier__tol': loguniform(1e-6, 1e-4),  # Corrected bounds for tolerance
        'classifier__class_weight': [None, 'balanced'],  # Class weighting strategies
        'classifier__max_iter': randint(50, 5001)  # Maximum iterations for solver
    },
    "DT": {
        'classifier__criterion': ['gini', 'entropy'],                # Splitting criterion
        'classifier__splitter': ['best', 'random'],                  # Split strategy at each node
        'classifier__max_depth': randint(3, 16),                      # Maximum depth of the tree
        'classifier__min_samples_split': randint(2, 10),              # Minimum samples to split a node
        'classifier__min_samples_leaf': randint(1, 10),               # Minimum samples in a leaf node
        'classifier__max_features': ['sqrt', 'log2', None],   # Number of features to consider at each split
        'classifier__max_leaf_nodes': randint(5, 50),                 # Maximum number of leaf nodes
        'classifier__class_weight': [None, 'balanced'],               # Handling class imbalance
    },
    "RF": {
        'classifier__n_estimators': randint(30, 300),             # Number of trees in the forest
        'classifier__criterion': ['gini', 'entropy'],               # Splitting criterion
        'classifier__max_depth': randint(3, 16),                     # Maximum depth of the tree
        'classifier__min_samples_split': randint(2, 10),             # Minimum samples required to split an internal node
        'classifier__min_samples_leaf': randint(1, 10),              # Minimum samples required at a leaf node
        'classifier__max_features': ['sqrt', 'log2', None],  # Number of features to consider for each split
        'classifier__max_leaf_nodes': randint(5, 50),                # Maximum number of leaf nodes
        'classifier__bootstrap': [True, False],                      # Whether to use bootstrap sampling
        'classifier__class_weight': [None, 'balanced'],              # Class weighting to handle class imbalance
    },
    "NN": {
        'classifier__hidden_layer_sizes': hidden_layer_sizes,  # Hidden layer architectures
        'classifier__activation': ['relu', 'tanh', 'logistic'],                     # Activation functions
        'classifier__solver': ['adam', 'sgd', 'lbfgs'],                             # Solvers for weight optimization
        'classifier__alpha': uniform(1e-5, 1e-1),                                   # Regularization term
        'classifier__batch_size': ['auto', 32, 64, 128],                            # Batch sizes
        'classifier__learning_rate': ['constant', 'invscaling', 'adaptive'],        # Learning rate schedule
        'classifier__learning_rate_init': uniform(1e-4, 1e-2),                      # Initial learning rate
        'classifier__max_iter': randint(100, 500),                                  # Maximum number of iterations
        'classifier__tol': uniform(1e-5, 1e-4),                                      # Tolerance for optimization
        'classifier__early_stopping': [True, False],                                 # Whether to use early stopping
        'classifier__validation_fraction': uniform(0.05, 0.2),                       # Fraction of training data for validation
        'classifier__n_iter_no_change': randint(5, 10),                              # Number of iterations with no improvement
    }
}

def printd(*args: Any, **kwargs: Any) -> None:
    """
    Prints the provided arguments. If the 'file' keyword argument is provided,
    it prints to the file and then to the default standard output.

    Args:
        *args: Positional arguments to be printed.
        **kwargs: Keyword arguments for the `print` function.
    """
    file: Optional[TextIO] = kwargs.get("file", None)

    # Print to the specified file if provided
    print(*args, **kwargs)

    # If 'file' is provided, remove it and print to default stdout
    if file is not None:
        del kwargs["file"]
        print(*args, **kwargs)


class Classifiers:
    def generate_init_models(self):
        init_config = {
            "LR": {
                "model": LogisticRegression,
                "config": self.config.get("classifiers").get("init_model_config").get("LR"),
            },
            "SVC": {
                "model": SVC,
                "config": self.config.get("classifiers").get("init_model_config").get("SVC"),
            },
            "DT": {
                "model": DecisionTreeClassifier,
                "config": self.config.get("classifiers").get("init_model_config").get("DT"),
            },
            "RF": {
                "model": RandomForestClassifier,
                "config": self.config.get("classifiers").get("init_model_config").get("RF"),
            },
            "NN": {
                "model": MLPClassifier,
                "config": self.config.get("classifiers").get("init_model_config").get("NN"),
            }
        }
        return init_config

    def generate_best_model_config(self):
        best_config = {}
        for model_name, model_info in self.init_models.items():
            best_config[model_name] = {
                "model": model_info.get("model"),
                "config": self.config.get("best_model_config").get(model_name).get("config").get("val_score", {}),
                "val_score": self.config.get("best_model_config").get(model_name).get("val_score", 0.0)
            }
        return best_config

    def __init__(self, config):
        self.config = config
        self.output_path = config.get("output").get("dir") + f"Classifier/"
        os.makedirs(self.output_path, exist_ok=True)
        self.file = open(self.output_path + "info.txt", "w")
        self.skfold = StratifiedKFold(n_splits=self.config.get("classifiers").get("k_folds"), shuffle=True, random_state=42)

        self.preprocess_pipelines = None
        self.eval_df = None
        self.agg_eval_df = None
        self.test_eval_df = None
        self.init_models = self.generate_init_models()
        self.best_model_config = self.generate_best_model_config()

    def __del__(self):
        self.file.close()

    def aggregate_performance(self, new_eval_df):
        grouped = new_eval_df.groupby(['preprocess_type', 'algo'])
        summary_data = []

        for (preprocess_type, algo), group in grouped:
            row = {
                'preprocess_type': preprocess_type,
                'algo': algo,
            }
            for col in self.eval_df.columns[2:]:
                mean = group[col].mean()
                std = group[col].std()
                row[col] = f"{mean:.4f} ±({3 * std:.4f})"
            summary_data.append(row)

        new_agg_eval_df = pd.DataFrame(summary_data).sort_values(by=['algo', 'preprocess_type'])
        if self.agg_eval_df is None:
            self.agg_eval_df = new_agg_eval_df
        else:
            self.agg_eval_df = pd.concat([self.agg_eval_df, new_agg_eval_df], axis=0, ignore_index=True).sort_values(
                by=['algo', 'preprocess_type'])

        return self.agg_eval_df


    def fit_and_eval(self, preprocess_pipelines, X_train, y_train):
        results = []

        for preprocess_type, p_pipeline in preprocess_pipelines.items():
            printd(f"Training {len(self.init_models)} models for {preprocess_type}: {self.init_models.keys()}", file=self.file)
            for algo, model_info in tqdm(self.init_models.items()):
                model_estimator_step = ("classifier", model_info.get("model")(**model_info.get("config")))
                pipeline_with_estimator = Pipeline(p_pipeline.steps + [model_estimator_step])
                algo_results = cross_validate(pipeline_with_estimator, X_train, y_train, cv=self.skfold,
                                              scoring=self.config.get("classifiers").get("scoring"),
                                              return_train_score=True, n_jobs=None)
                algo_results_df = pd.DataFrame(algo_results)
                algo_results_df.insert(0, 'preprocess_type', preprocess_type)
                algo_results_df.insert(1, 'algo', algo)
                algo_results_df.columns = [col.replace('test_', 'val_') for col in algo_results_df.columns]
                results.append(algo_results_df)

        new_eval_df = pd.concat(results, ignore_index=True).sort_values(by=['algo', 'preprocess_type'])
        if self.eval_df is None:
            self.eval_df = new_eval_df
        else:
            self.eval_df = pd.concat([self.eval_df, new_eval_df], axis=0, ignore_index=True).sort_values(
                by=['algo', 'preprocess_type'])

        print(f"Evaluation Results:", file=self.file)
        print(self.eval_df, file=self.file)
        self.aggregate_performance(new_eval_df)
        print(f"Aggregate Results:", file=self.file)
        print(self.agg_eval_df, file=self.file)

        return self.eval_df, self.agg_eval_df

    def fit_and_eval_test_set(self, preprocess_pipelines, X_train, y_train, X_test, y_test):
        results = []

        for preprocess_type, p_pipeline in preprocess_pipelines.items():
            printd(f"Training {len(self.init_models)} models for {preprocess_type}: {self.init_models.keys()} on full train set", file=self.file)
            for algo, model_info in tqdm(self.best_model_config.items()):
                model_estimator_steps = ("classifier", model_info.get("model")(**model_info.get("config")))

                pipeline_with_estimator = Pipeline(p_pipeline.steps + [model_estimator_steps])
                pipeline_with_estimator.fit(X_train, y_train)
                # Predictions on train and test sets
                y_train_pred = pipeline_with_estimator.predict(X_train)
                y_test_pred = pipeline_with_estimator.predict(X_test)

                # Compute metrics
                algo_results = {
                    "test_accuracy": accuracy_score(y_test, y_test_pred),
                    "train_accuracy": accuracy_score(y_train, y_train_pred),
                    "test_precision": precision_score(y_test, y_test_pred),
                    "train_precision": precision_score(y_train, y_train_pred,),
                    "test_recall": recall_score(y_test, y_test_pred),
                    "train_recall": recall_score(y_train, y_train_pred),
                    "test_f1": f1_score(y_test, y_test_pred),
                    "train_f1": f1_score(y_train, y_train_pred),
                }

                # Convert metrics to a single row (dictionary or pandas DataFrame)
                algo_results_df = pd.DataFrame([algo_results])
                algo_results_df.insert(0, 'preprocess_type', preprocess_type)
                algo_results_df.insert(1, 'algo', algo)
                results.append(algo_results_df)

        if self.test_eval_df is None:
            self.test_eval_df = pd.concat(results, ignore_index=True).sort_values(by=['algo', 'preprocess_type'])
        else:
            new_test_eval_df = pd.concat(results, ignore_index=True)
            self.test_eval_df = pd.concat([self.test_eval_df, new_test_eval_df], axis=0, ignore_index=True).sort_values(
                by=['algo', 'preprocess_type'])

        print(f"Test Evaluation Results:", file=self.file)
        print(self.test_eval_df, file=self.file)

        return self.test_eval_df

    def check_and_update_best_config(self, algo, best_params, best_score, file_path="config.json"):
        """
        Check and update the best configuration for an algorithm and save it to a file.

        Parameters:
            algo (str): The algorithm name.
            best_params (dict): The best parameters for the algorithm.
            best_score (float): The best score achieved with the new parameters.
            file_path (str): The path to the JSON file to save the updated config.
        """
        old_score = self.config.get("best_model_config").get(algo, {}).get("best_score_", float("-inf"))
        if best_score > old_score:
            # Clean the parameters (convert np.float64 to float)
            cleaned_config = {key: float(value) if isinstance(value, np.float64) else value for key, value in
                              best_params.items()}

            # Update the config dictionary
            self.config["best_model_config"][algo] = {
                "config": cleaned_config,
                "best_score_": best_score
            }

            # Save the updated config to the specified file
            with open(file_path, "w") as json_file:
                json.dump(self.config, json_file, indent=4)

            printd(f"Updated configuration for {algo} saved to {file_path}", file=self.file)
        else:
            printd(
                f"New score for {algo} ({best_score}) is not better than the old score ({old_score}). No update made.",
            file=self.file)

    def plot_val_metric_comp(self):
        metrics = ['val_precision', 'val_recall', 'val_f1']
        def extract_mean_and_uncertainty(value):
            mean, uncertainty = value.split(' ±')
            return float(mean), float(uncertainty.strip('()'))

        # Split mean and uncertainty into separate columns
        for col in metrics:
            self.agg_eval_df[[f'{col}_mean', f'{col}_uncertainty']] = self.agg_eval_df[col].apply(
                lambda x: pd.Series(extract_mean_and_uncertainty(x)))

        x = np.arange(len(self.agg_eval_df['algo'].unique()))  # Position of bars for each group
        width = 1 / (len(self.agg_eval_df['preprocess_type'].unique()) + 1.0)

        for metric in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))

            for i, ptype in enumerate(self.agg_eval_df['preprocess_type'].unique()):
                subset = self.agg_eval_df[self.agg_eval_df['preprocess_type'] == ptype]
                means = subset[f'{metric}_mean']
                uncertainties = subset[f'{metric}_uncertainty']
                ax.bar(
                    x + i * width,
                    means,
                    width,
                    label=ptype,
                    # color=colors[i],
                    yerr=uncertainties,
                    capsize=5  # Add caps to error bars
                )

            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(self.agg_eval_df['algo'].unique())
            ax.set_xlabel('Algorithms')
            ax.set_ylabel(metric.replace('_', ' ').capitalize())
            ax.set_title(f'Comparison of {metric.replace("_", " ").capitalize()} with Uncertainty')
            ax.legend(title='Preprocessing Type')
            plt.tight_layout()
            plt.show()

    def plot_test_metric_comp(self):
        # List of metrics to plot
        metrics = ['test_precision', 'test_recall', 'test_f1']

        # Create subplots for each metric
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

        # Colors for the bars
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        # Loop through each metric to plot
        for i, metric in enumerate(metrics):
            ax = axes[i]
            ax.bar(self.test_eval_df['algo'], self.test_eval_df[metric], color=colors, alpha=0.8)
            ax.set_title(f'Comparison of {metric.replace("_", " ").capitalize()}')
            ax.set_xlabel('Algorithm')
            ax.set_ylabel(metric.replace('_', ' ').capitalize())
            ax.set_ylim(0, 1)  # Assuming metric values are between 0 and 1

        # Adjust layout
        plt.tight_layout()
        plt.show()

    def hyper_parm_tuning(self, preprocess_pipeline, X_train, y_train, algos=None, n_iter=100):
        if algos is None:
            algos = self.init_models.keys()

        for algo in algos:
            printd(f"Tuning  {algo} for  {n_iter} iteration",file=self.file)
            model_estimator_steps = (
            "classifier", self.init_models.get(algo).get("model")(**self.init_models.get(algo).get("config")))
            pipeline_with_estimator = Pipeline(preprocess_pipeline.steps + [model_estimator_steps])
            random_search = RandomizedSearchCV(pipeline_with_estimator, param_distributions=random_param_dist.get(algo),
                                               n_iter=n_iter, cv=self.config.get("classifiers").get("k_folds"),
                                               scoring="f1",
                                               n_jobs=-1)
            random_search.fit(X_train, y_train)

            # Best parameters and score
            printd("Best parameters:", random_search.best_params_, file=self.file)
            printd("Best cross-validation score:", random_search.best_score_, file=self.file)
            printd("-"*100, file=self.file)

            # check and update best param if true
            self.check_and_update_best_config(algo, random_search.best_params_, random_search.best_score_)



if __name__ == "__main__":
    import json
    from preprocess import Preprocess, download_data, split_train_test
    config_path = "./config.json"
    with open(config_path, "r") as file:
        config = json.load(file)

    raw_df = download_data(config, n_rows=None)
    X_train, X_test, y_train, y_test  = split_train_test(raw_df, config)

    p = Preprocess(config)
    baseline_pipeline = p.gen_basic_pipeline()

    c  = Classifiers(config)
    k_fold_eval, agg_eval = c.fit_and_eval({"baseline": baseline_pipeline}, X_train, y_train)
    print(agg_eval)