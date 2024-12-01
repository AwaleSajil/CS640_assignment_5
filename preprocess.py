import json
import os
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class AgeBinner(BaseEstimator, TransformerMixin):
    def __init__(self, age_col="age", num_bins=11, min_age=17, max_age=98):
        self.age_col = age_col
        self.num_bins = num_bins
        self.min_age = min_age
        self.max_age = max_age

        self.binned_columns_ = []  # Store generated columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        bins = np.linspace(self.min_age, self.max_age, self.num_bins + 1)
        labels = [
            f"{int(bins[i])}_{int(bins[i + 1]) - 1}" for i in range(self.num_bins)
        ]

        binned = pd.cut(X, bins=bins, labels=labels, right=False)
        age_binned_encoded = pd.get_dummies(binned, prefix="age_", dtype=float)

        # Update the list of generated columns
        self.binned_columns_ = age_binned_encoded.columns.tolist()

        return age_binned_encoded

    def get_params(self, deep=True):
        return {
            "age_col": self.age_col,
            "num_bins": self.num_bins,
            "min_age": self.min_age,
            "max_age": self.max_age,
        }

    def get_feature_names_out(self, input_features=None):
        # Return the dynamically generated column names
        return self.binned_columns_


class PdaysBinner(BaseEstimator, TransformerMixin):
    def __init__(self, pday_col="pdays", num_bins=3, min_value=0, max_value=26):
        self.pday_col = pday_col
        self.num_bins = num_bins
        self.min_value = min_value
        self.max_value = max_value
        self.binned_columns_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Make a copy of the Series to avoid modifying the original
        X_copy = X.copy()

        # Bin the pdays feature
        bins = np.linspace(self.min_value, self.max_value, self.num_bins + 1)
        labels = [
            f"{int(bins[i])}_{int(bins[i + 1]) - 1}" for i in range(self.num_bins)
        ]

        # Create binned categories, using the clipped values
        binned = pd.cut(
            X_copy.clip(upper=self.max_value),
            bins=bins,
            labels=labels,
            right=False,
        )
        binned_encoded = pd.get_dummies(binned, prefix="pdays", dtype=float)

        # Feature for pdays > max bin value
        binned_encoded[f"pdays_gt_{int(bins[-1] - 1)}"] = (
            X_copy > (bins[-1] - 1)
        ).astype(float)
        never_contacted = (X_copy == 999).astype(float)
        binned_encoded[never_contacted == 1] = 0
        binned_encoded["pdays_never_contacted"] = never_contacted

        # Update the list of generated columns
        self.binned_columns_ = list(binned_encoded.columns)

        # Return the transformed data
        return binned_encoded

    def get_params(self, deep=True):
        return {
            "pday_col": self.pday_col,
            "num_bins": self.num_bins,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }

    def get_feature_names_out(self, input_features=None):
        # Return the dynamically generated column names
        return self.binned_columns_


class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_drop):
        self.features_to_drop = features_to_drop

    def fit(self, X, y=None):
        return self  # No fitting required, just return self

    def transform(self, X):
        # Drop the specified features from the DataFrame
        X_dropped = X.drop(
            columns=[col for col in self.features_to_drop if col in X.columns],
            axis=1,
        )
        return X_dropped


class Preprocess:
    def __init__(self, config, df):
        self.config = config
        self.df = df
        self.target_col = self.config.get("preprocess").get("target_col")
        self.output_path = config.get("output").get("dir") + "Preprocess/"
        os.makedirs(self.output_path, exist_ok=True)
        self.file = open(self.output_path + "info.txt", "w")

        self.continuous_features = self.df.select_dtypes(
            include=["int64", "float64"],
        ).columns
        self.categorical_features = self.df.select_dtypes(include=["object"]).columns

    def __del__(self):
        self.file.close()

    def plot_distribution(self, raw_df, split):
        plot_dir = f"{self.output_path}{split}/distribution_plots/"
        os.makedirs(plot_dir, exist_ok=True)

        # Loop through continuous features and plot KDE for each
        for feature in self.continuous_features:
            # Create a 1x2 grid of subplots (1 row, 2 columns)
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))

            # KDE Plot on the first subplot (axes[0])
            sns.kdeplot(
                data=raw_df[raw_df[self.target_col] == 1],
                x=feature,
                label="Target: Yes",
                fill=True,
                alpha=0.5,
                ax=axes[0],
            )

            sns.kdeplot(
                data=raw_df[raw_df[self.target_col] == 0],
                x=feature,
                label="Target: No",
                fill=True,
                alpha=0.5,
                ax=axes[0],
            )
            axes[0].set_title(f"Density Plot of {feature} by Target")
            axes[0].set_xlabel(feature)
            axes[0].set_ylabel("Density")
            axes[0].legend()

            # Box Plot on the second subplot (axes[1])
            sns.boxplot(
                x=self.target_col,
                y=feature,
                data=raw_df,
                palette="Set2",
                ax=axes[1],
                hue=self.target_col,
            )
            axes[1].set_title(f"Boxplot of {feature} by Target")
            axes[1].set_xlabel(self.target_col)
            axes[1].set_ylabel(feature)

            # Adjust layout to make sure plots don't overlap
            plt.tight_layout()

            # Save the combined plot to a file
            plt.savefig(f"{plot_dir}{feature}.png")
            plt.close(fig)

        # Loop through categorical features
        for feature in self.categorical_features:
            # Check if feature exists and has valid data
            if feature not in raw_df.columns or raw_df[feature].isna().all():
                print(f"Skipping {feature}: No valid data.")
                continue

            # Calculate proportions
            proportions = (
                raw_df.groupby(feature)[self.target_col]
                .value_counts(normalize=True)
                .unstack()
            )

            # Create figure and axes explicitly
            fig, ax = plt.subplots(figsize=(8, 5))

            # Plot stacked bar plot
            proportions.plot(kind="bar", stacked=True, alpha=0.75, ax=ax)

            # Set titles and labels
            ax.set_title(f"Proportion of Target by {feature} for {split} set")
            ax.set_xlabel(feature)
            ax.set_ylabel("Proportion")
            ax.legend(title="Target", loc="upper right")
            plt.xticks(rotation=45)

            # Adjust layout
            plt.tight_layout()

            # Save the plot
            plot_file = (
                f"{plot_dir}/{feature}.png"
                if plot_dir.endswith("/")
                else f"{plot_dir}{feature}.png"
            )
            plt.savefig(plot_file)

            # Close the figure
            plt.close(fig)

    def generate_raw_data_info(self, raw_df, split):
        self.plot_distribution(raw_df, split)
        print(f"Initial shape of data: {raw_df.shape}", file=self.file)
        # finding label count
        print("Class label count: ", file=self.file)
        print(
            raw_df[self.config.get("preprocess").get("target_col")].value_counts(),
            file=self.file,
        )
        # adding unique values to info
        unique_values = {
            col: (raw_df[col].value_counts(normalize=True).to_dict())
            for col in raw_df.columns
        }
        print("Unique values of each columns: ", file=self.file)
        print(json.dumps(unique_values, indent=4), file=self.file)

    def _get_feature_names(self, preprocessor):
        # Get the feature names after one-hot encoding
        onehot_columns = preprocessor.named_transformers_["cat_features"][
            "onehot_encoder"
        ].get_feature_names_out(
            self.categorical_features,
        )
        age_bin_columns = preprocessor.named_transformers_["age"][
            "age_binner"
        ].get_feature_names_out()
        cont_features_without_age = [
            i
            for i in self.continuous_features
            if i not in [self.config.get("preprocess").get("age_bin").get("col")]
        ]
        return list(onehot_columns) + cont_features_without_age + list(age_bin_columns)

    def gen_basic_pipeline(self, features_to_ignore=None):
        if features_to_ignore is None:
            features_to_ignore = []
        categorical_features = [
            i for i in self.categorical_features if i not in features_to_ignore
        ]
        continuous_features = [
            i for i in self.continuous_features if i not in features_to_ignore
        ]

        # impute categorical values with unknown
        cat_imputer = SimpleImputer(strategy="most_frequent", missing_values="unknown")
        # encode categorical features
        cat_dummies = OneHotEncoder(handle_unknown="ignore")

        categorical_pipeline = Pipeline(
            [
                ("cat_imputer", cat_imputer),
                ("onehot_encoder", cat_dummies),
            ],
        )

        continuous_pipeline = Pipeline(
            [
                (
                    "passthrough",
                    "passthrough",
                ),  # No transformation for numerical columns
            ],
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat_features", categorical_pipeline, categorical_features),
                (
                    "cont_features",
                    continuous_pipeline,
                    [
                        i
                        for i in continuous_features
                        if i
                        not in [
                            self.config.get("preprocess").get("age_bin").get("col"),
                            "pdays",
                        ]
                    ],
                ),
            ],
        )

        full_pipeline = Pipeline(
            [
                (
                    "preprocessing",
                    preprocessor,
                ),  # Categorical and numerical preprocessing
                (
                    "scaling",
                    StandardScaler(),
                ),  # Apply scaling to all features at the end
            ],
        )

        return full_pipeline

    def gen_pipeline_with_age_pdays_binned(self, features_to_ignore=None):
        # Start with the basic pipeline as the baseline
        full_pipeline = self.gen_basic_pipeline(features_to_ignore)

        # Extract the preprocessor from the baseline pipeline
        preprocessor = full_pipeline.named_steps["preprocessing"]

        # Add custom transformers for age binning and pdays binning
        age_binning_pipeline = Pipeline(
            [
                (
                    "age_binner",
                    AgeBinner(
                        age_col=self.config.get("preprocess").get("age_bin").get("col"),
                        num_bins=self.config.get("preprocess")
                        .get("age_bin")
                        .get("num_bins", 11),
                        min_age=self.config.get("preprocess")
                        .get("age_bin")
                        .get("min_age", 17),
                        max_age=self.config.get("preprocess")
                        .get("age_bin")
                        .get("max_age", 98),
                    ),
                ),
            ],
        )

        pdays_binning_pipeline = Pipeline(
            [
                ("pdays_binner", PdaysBinner(pday_col="pdays")),
            ],
        )

        # Update the column transformer with the additional transformations
        new_transformers = preprocessor.transformers + [
            (
                "age",
                age_binning_pipeline,
                self.config.get("preprocess").get("age_bin").get("col"),
            ),
            ("pdays", pdays_binning_pipeline, "pdays"),
        ]

        # Create a new ColumnTransformer with the updated transformers
        updated_preprocessor = ColumnTransformer(transformers=new_transformers)

        # Replace the preprocessor in the full pipeline
        full_pipeline.steps[0] = ("preprocessing", updated_preprocessor)

        return full_pipeline

    def gen_pipeline_with_feature_drop(self, features_to_drop=None):
        # Default to empty list if no features are provided to drop
        features_to_drop = features_to_drop or []

        # Create a feature dropping step
        drop_features_step = ("drop_features", DropFeatures(features_to_drop))

        # Get the remaining pipeline from the basic pipeline generator
        basic_pipeline = self.gen_pipeline_with_age_pdays_binned(features_to_drop).steps

        # Combine all steps into a single pipeline
        full_pipeline_with_drop = Pipeline([drop_features_step] + basic_pipeline)

        return full_pipeline_with_drop

    def gen_pipeline_with_pca(self, pca_variance_to_keep, features_to_drop=None):
        # Get the pipeline with feature dropping from gen_pipeline_with_feature_drop
        feature_drop_pipeline = self.gen_pipeline_with_feature_drop(
            features_to_drop,
        ).steps

        # Add the PCA step after the feature dropping step
        pca_step = ("pca", PCA(n_components=pca_variance_to_keep, random_state=42))

        # Combine all steps into a single pipeline
        full_pipeline_with_pca = Pipeline(feature_drop_pipeline + [pca_step])

        return full_pipeline_with_pca

    def gen_pipeline_with_smote(self, pca_variance_to_keep, features_to_drop=None):
        prev_pipeline = self.gen_pipeline_with_feature_drop(features_to_drop).steps
        smote_step = ("smote", SMOTE(sampling_strategy="auto", random_state=42))
        new_pipeline = Pipeline(prev_pipeline + [smote_step])

        if pca_variance_to_keep:
            pca_step = self.gen_pipeline_with_pca(
                pca_variance_to_keep,
                features_to_drop=None,
            ).steps[-1]
            new_pipeline = Pipeline(new_pipeline.steps + [pca_step])

        return new_pipeline

    def gen_pipeline_with_smoteenn(self, pca_variance_to_keep, features_to_drop=None):
        prev_pipeline = self.gen_pipeline_with_feature_drop(features_to_drop).steps
        smoteenn_step = (
            "smoteenn",
            SMOTEENN(sampling_strategy="auto", random_state=42),
        )
        new_pipeline = Pipeline(prev_pipeline + [smoteenn_step])

        if pca_variance_to_keep:
            pca_step = self.gen_pipeline_with_pca(
                pca_variance_to_keep,
                features_to_drop=None,
            ).steps[-1]
            new_pipeline = Pipeline(new_pipeline.steps + [pca_step])

        return new_pipeline


def download_data(config, n_rows=None):
    local_path = config.get("preprocess").get("data_local_path")
    os.makedirs(local_path, exist_ok=True)
    urllib.request.urlretrieve(
        config.get("preprocess").get("data_url"),
        local_path + "data.zip",
    )
    with zipfile.ZipFile(local_path + "data.zip", "r") as zip_ref:
        zip_ref.extractall(local_path)

    raw_df = pd.read_csv(
        local_path + config.get("preprocess").get("zip_file_struct"),
        sep=config.get("preprocess").get("data_sep"),
    )
    if n_rows:
        raw_df = raw_df.head(n_rows)

    target_col = config.get("preprocess").get("target_col")
    raw_df[target_col] = raw_df[target_col].map({"yes": 1, "no": 0})
    raw_df = raw_df.drop_duplicates()
    return raw_df


def split_train_test(raw_df, config):
    target_col = config.get("preprocess").get("target_col")
    X = raw_df.drop(columns=[target_col])
    y = raw_df[target_col].values
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.get("preprocess").get("train_test_split"),
        random_state=42,
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    config_path = "./config.json"
    with open(config_path, "r") as file:
        config = json.load(file)

    raw_df = download_data(config, n_rows=None)
    X_train, X_test, y_train, y_test = split_train_test(raw_df, config)

    p = Preprocess(config)
    baseline_pipeline = p.gen_basic_pipeline()
