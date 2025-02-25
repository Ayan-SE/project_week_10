import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
#import ruptures as rpt
#import pymc3 as pm

def plot_data(df, x_col, y_cols, plot_type='line', title=None, xlabel=None, ylabel=None, figsize=(10, 6), **kwargs):
    """
    Plots data from a pandas DataFrame.
    """

    plt.figure(figsize=figsize)  # Create the figure

    if isinstance(y_cols, str):  # If y_cols is a single string
        y_cols = [y_cols]  # Make it a list

    if plot_type == 'line':
        for y_col in y_cols:
            plt.plot(df[x_col], df[y_col], label=y_col, **kwargs)
    elif plot_type == 'scatter':
        for y_col in y_cols:
            plt.scatter(df[x_col], df[y_col], label=y_col, **kwargs)
    elif plot_type == 'bar':
        for y_col in y_cols:
            plt.bar(df[x_col], df[y_col], label=y_col, **kwargs)
    elif plot_type == 'hist':
        for y_col in y_cols:
            plt.hist(df[y_col], label=y_col, **kwargs)  # x_col is not used for histograms
    elif plot_type == 'box':
        plt.boxplot([df[y_col] for y_col in y_cols], labels=y_cols, **kwargs)  # x_col not directly used
    else:
        raise ValueError("Invalid plot_type. Choose from 'line', 'scatter', 'bar', 'hist', 'box'.")

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    if len(y_cols) > 1 and plot_type != 'box': # Add legend if multiple y-columns (not for boxplots)
        plt.legend()

    plt.grid(True)  # Add a grid for better readability
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()

    # Return the axes object(s) for further customization if needed
    if len(y_cols) == 1 or plot_type == 'box':
        return plt.gca()  # Return the current Axes
    else:
        return plt.gcf().get_axes()  # Return a list of Axes
    
def cusum_method(data, k, h, initial_mean=None, title="CUSUM Chart"):
    """
    Perform CUSUM analysis to detect change points in a dataset.
    """
    
    # Ensure data is in a pandas Series
    if isinstance(data, pd.Series):
        data = data.values
    
    # Calculate the mean if not provided
    if initial_mean is None:
        initial_mean = np.mean(data[:min(10, len(data))])  # use the first 10 or fewer values
    
    # Initialize arrays for the positive and negative CUSUM values
    n = len(data)
    cusum_positive = np.zeros(n)
    cusum_negative = np.zeros(n)
    change_points = []

    # Perform CUSUM analysis
    for i in range(1, n):
        cusum_positive[i] = max(0, cusum_positive[i - 1] + data[i] - initial_mean - k)
        cusum_negative[i] = min(0, cusum_negative[i - 1] + data[i] - initial_mean + k)

        # Detect change points based on the threshold 'h'
        if cusum_positive[i] > h or cusum_negative[i] < -h:
            change_points.append(i)
            cusum_positive[i] = 0  # Reset CUSUMs (optional)
            cusum_negative[i] = 0  # Reset CUSUMs (optional)
    
    # Plot the CUSUM chart
    plt.figure(figsize=(10, 6))
    plt.plot(cusum_positive, label="CUSUM Positive", color="green")
    plt.plot(cusum_negative, label="CUSUM Negative", color="red")
    plt.axhline(h, color='blue', linestyle='--', label="Upper Threshold")
    plt.axhline(-h, color='orange', linestyle='--', label="Lower Threshold")
    
    # Highlight the change points
    for point in change_points:
        plt.axvline(x=point, color="black", linestyle=":", label="Change Point")
    
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("CUSUM Value")
    plt.legend(loc="best")
    plt.show()
    
    return change_points

"""
def bayesian_change_point_detection(data, model_type="Normal", num_samples=1000):
   
    # Ensure data is in a pandas Series
    if isinstance(data, pd.Series):
        data = data.values
    
    n = len(data)
    
    # Model Setup
    with pm.Model() as model:
        # Prior for the number of change points
        num_change_points = pm.DiscreteUniform("num_change_points", lower=1, upper=n-1)
        
        # Assign prior to the change points locations
        change_points = pm.Uniform("change_points", lower=0, upper=n, shape=(num_change_points,))
        
        # Prior distributions for the segments
        segment_means = pm.Normal("segment_means", mu=0, sigma=10, shape=(num_change_points+1,))
        segment_stds = pm.HalfNormal("segment_stds", sigma=10, shape=(num_change_points+1,))
        
        # Likelihood: Data follows a mixture of normal distributions for each segment
        observed_data = data[:1]
        
        for i in range(1, n):
            segment_index = pm.math.sum(i > change_points)
            observed_data = pm.Normal(f"obs_{i}", mu=segment_means[segment_index], sigma=segment_stds[segment_index], observed=data[i])
        
        # Sampling from the posterior
        trace = pm.sample(num_samples, chains=2, return_inferencedata=False)
        
    # Extracting posterior samples of change points
    change_points_posterior = trace["change_points"]
    
    # Compute the detected change points from posterior samples
    detected_change_points = []
    for point in change_points_posterior:
        detected_change_points.extend(point)
        
    # Plot the change points
    plt.figure(figsize=(10, 6))
    plt.plot(data, label="Data")
    for point in detected_change_points:
        plt.axvline(x=point, color="red", linestyle="--", label="Change Point")
    
    plt.title("Bayesian Change Point Detection")
    plt.xlabel("Index")
    plt.ylabel("Data Value")
    plt.legend()
    plt.show()
    
    return detected_change_points
"""

def plot_time_series(df, column, title):
    """
    Plots a time series graph for a specific column.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[column], label=column, color="blue")
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(column)
    plt.legend()
    plt.grid(True)
    plt.show()

# ------------------ 5. Identify Trends (Rolling Average) ------------------
def rolling_average(df, column, window=30):
    """
    Computes rolling average to identify trends.
.
    """
    df[f"{column}_rolling"] = df[column].rolling(window=window).mean()
    return df

# ------------------ 6. Correlation Analysis ------------------
def plot_correlation_heatmap(df):
    """
    Plots a correlation heatmap for numerical columns in a DataFrame.
    """
    numerical_df = df.select_dtypes(include=np.number) #Keep only numerical columns
    plt.figure(figsize=(10, 6))
    sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()

# ------------------ 7. Outlier Detection ------------------
def detect_outliers(df, column):
    """
    Detects outliers in a given column using the IQR method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Removing Outliers
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    print(f"✅ Outliers removed from {column}. Original: {len(df)}, New: {len(df_filtered)}")
    
    return df_filtered