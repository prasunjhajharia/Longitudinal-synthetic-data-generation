# Load & preprocess time series data for Dynamic Bayesian Network (DBN) modeling

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    def __init__(self, filepath):
        """
        Initializes the DataLoader with the path to the CSV file.

        Args:
            filepath (str): Path to the healthcare dataset CSV file.
        """
        self.filepath = filepath
        self.encoders = {}  # Dictionary to store label encoders for each categorical feature

    def load_data(self):
        """
        Loads the raw healthcare event-based data from a CSV file.

        Returns:
            pd.DataFrame: The raw dataset as a DataFrame.
        """
        df = pd.read_csv(self.filepath)
        return df

    def preprocess_health_dataset(self, df):
        """
        Preprocesses the raw healthcare dataset:
        - Excludes the 'subject_id' column from features
        - Label-encodes all categorical variables
        - Groups rows by 'subject_id' to form individual time series per subject
        - Concatenates all individual time series into a single DataFrame

        Args:
            df (pd.DataFrame): Raw healthcare dataset

        Returns:
            stacked_df (pd.DataFrame): All subjects' time series stacked together
            subject_lengths (dict): A mapping from subject_id to their individual time series length
            label_encoders (dict): Dictionary of LabelEncoders used per column
        """
        # Exclude 'subject_id' from features, but keep it temporarily for grouping
        features = [col for col in df.columns if col != 'subject_id']
        df = df[['subject_id'] + features].copy()

        label_encoders = {}

        # Label encode each categorical feature column
        for col in features:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))  # Convert to string to handle mixed types
            label_encoders[col] = le

        # Sort by subject_id to ensure consistent ordering
        df = df.sort_values(by='subject_id').reset_index(drop=True)

        # Group by subject_id to form per-patient time series
        df_grouped = df.groupby('subject_id')
        time_series_list = []      # List of time series per subject
        subject_lengths = {}       # Mapping of subject_id to their time series length

        for subject_id, group in df_grouped:
            group = group.drop(columns=['subject_id']).reset_index(drop=True)  # Remove subject_id from data
            time_series_list.append(group)
            subject_lengths[subject_id] = len(group)

        # Stack all subject time series into one multivariate time series
        stacked_df = pd.concat(time_series_list, axis=0, ignore_index=True)
        self.encoders = label_encoders  # Save encoders for later decoding
        return stacked_df, subject_lengths, label_encoders

    def create_lagged_matrix(self, df):
        """
        Constructs lagged feature matrix for DBN training.
        Creates (X_t, X_{t-1}) pairs by shifting the data.

        Args:
            df (pd.DataFrame): A multivariate time series (stacked from multiple subjects)

        Returns:
            X_t (pd.DataFrame): Features at time t
            X_lag (pd.DataFrame): Features at time t-1
        """
        # Create target and lagged input frames
        X_t = df.iloc[1:].reset_index(drop=True)              # Current timestep (t)
        X_lag = df.shift(1).iloc[1:].reset_index(drop=True)   # Previous timestep (t-1)

        return X_t, X_lag
