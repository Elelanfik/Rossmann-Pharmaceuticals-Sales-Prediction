import pandas as pd
import numpy as np
import pickle
import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


class ModelBuilder:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.X_train = None
        self.y_train = None

    def split_data(self, target_column):
        """
        Splits the dataset into features (X) and target (y).
        Args:
            target_column (str): The name of the target column to predict.
        
        Returns:
            X (pd.DataFrame): The feature variables.
            y (pd.Series): The target variable.
        """
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        return X, y

    def build_random_forest(self):
        
        X, y = self.split_data(target_column='Sales')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Save the training data in class variables
        self.X_train = X_train
        self.y_train = y_train
        # Model Pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(n_estimators=50, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        self.model = pipeline
        
        # Evaluate model
        y_pred = pipeline.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f'Mean Squared Error (MSE): {mse}')
        print(f'Root Mean Squared Error (RMSE): {rmse}')
        print(f'Mean Absolute Error (MAE): {mae}')
        print(f'R-Squared (R2): {r2}')
        
        # Plot Predicted vs True
        self.plot_predicted_vs_true(y_test, y_pred)

        # Plot Evaluation Metrics
        self.plot_evaluation_metrics_normalized(mse, rmse, mae, r2)
        
        return pipeline, mse, rmse, mae, r2

    def feature_importance(self, plot=True):
        """
        Extracts and returns feature importance from the Random Forest model.
        
        If plot=True, displays a bar plot of feature importance.

        Args:
            plot (bool): If True, plot the feature importances.

        Returns:
            pd.DataFrame: Sorted feature importance or None if unavailable.
        """
        if hasattr(self.model.named_steps['rf'], 'feature_importances_'):
            importances = self.model.named_steps['rf'].feature_importances_
            features = self.X_train.columns  # Access features from saved X_train
            importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            
            if plot:
                # Plotting feature importances
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=importance_df)
                plt.title('Feature Importance')
                plt.tight_layout()
                plt.show()
            
            return importance_df
        else:
            print("No feature importances available.")
            return None

    def serialize_model(self):
        """Serialize the model to a file with a timestamp in the filename."""
        # Get the current timestamp
        timestamp = datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        filename = f"random_forest_model-{timestamp}.pkl"

        # Serialize the model
        with open(f"../model/{filename}", 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"Model saved as {filename}")
        return filename

    def plot_predicted_vs_true(self, y_test, y_pred):
        """Plots predicted vs. true values."""
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title("Predicted vs. True Values")
        plt.tight_layout()
        plt.show()

    def plot_evaluation_metrics_normalized(self, mse, rmse, mae, r2):
        metrics = ['MSE', 'RMSE', 'MAE', 'R2']
        values = [mse, rmse, mae, r2]

        # Normalize values to the range [0, 1]
        max_value = max(values)
        normalized_values = [v / max_value for v in values]

        plt.figure(figsize=(8, 6))
        sns.barplot(x=metrics, y=normalized_values, palette='viridis')
        plt.title("Normalized Model Evaluation Metrics")
        plt.ylabel("Normalized Metric Value")
        plt.tight_layout()
        plt.show()
