import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def classify_holiday_periods(df):
    """
    Classify each date into Before Holiday, During Holiday, or After Holiday.
    Args:
        df (DataFrame): Dataset with 'Date', 'StateHoliday', and 'SchoolHoliday' columns.
    Returns:
        DataFrame: Updated DataFrame with a new column 'HolidayPeriod'.
    """
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Identify holiday dates
    df['IsHoliday'] = np.where((df['StateHoliday'] != '0') | (df['SchoolHoliday'] == 1), 1, 0)
    df['HolidayPeriod'] = 'No Holiday'

    # Classify periods
    for idx in df.index:
        if df.loc[idx, 'IsHoliday'] == 1:
            df.loc[idx, 'HolidayPeriod'] = 'During Holiday'
        elif idx > 0 and df.loc[idx - 1, 'IsHoliday'] == 1:
            df.loc[idx, 'HolidayPeriod'] = 'After Holiday'
        elif idx < len(df) - 1 and df.loc[idx + 1, 'IsHoliday'] == 1:
            df.loc[idx, 'HolidayPeriod'] = 'Before Holiday'

    logging.info("Classified holiday periods as Before, During, and After holidays.")
    return df


def visualize_holiday_sales_behavior_all(df):
    """
    Visualize and compare sales behavior before, during, and after holidays using multiple plots.
    Args:
        df (DataFrame): Dataset with 'Sales' and 'HolidayPeriod' columns.
    """
    holiday_order = ['Before Holiday', 'During Holiday', 'After Holiday', 'No Holiday']
    
    # Boxplot for Sales Behavior
    plt.figure(figsize=(14, 7))
    sns.boxplot(data=df, x='HolidayPeriod', y='Sales', order=holiday_order, palette="Set2")
    plt.title("Sales Behavior Before, During, and After Holidays (Boxplot)", fontsize=16)
    plt.xlabel("Holiday Period", fontsize=14)
    plt.ylabel("Sales", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    logging.info("Boxplot visualized for sales behavior across holiday periods.")
    
    # Violin Plot for Distribution Comparison
    plt.figure(figsize=(14, 7))
    sns.violinplot(data=df, x='HolidayPeriod', y='Sales', order=holiday_order, palette="coolwarm", split=True)
    plt.title("Sales Behavior Before, During, and After Holidays (Violin Plot)", fontsize=16)
    plt.xlabel("Holiday Period", fontsize=14)
    plt.ylabel("Sales", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    logging.info("Violin plot visualized for sales behavior across holiday periods.")

    # Bar Plot for Mean Sales
    avg_sales = df.groupby('HolidayPeriod')['Sales'].mean().reindex(holiday_order)
    plt.figure(figsize=(12, 6))
    avg_sales.plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    plt.title("Average Sales Before, During, and After Holidays (Bar Plot)", fontsize=16)
    plt.ylabel("Average Sales", fontsize=14)
    plt.xlabel("Holiday Period", fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    logging.info("Bar plot visualized for average sales across holiday periods.")

    # Line Plot for Trend Over Time
    df['Date'] = pd.to_datetime(df['Date'])
    plt.figure(figsize=(14, 7))
    for period in holiday_order:
        subset = df[df['HolidayPeriod'] == period]
        plt.plot(subset['Date'], subset['Sales'], label=period, alpha=0.8)
    plt.title("Sales Trends Over Time by Holiday Period (Line Plot)", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Sales", fontsize=14)
    plt.legend(title="Holiday Period")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    logging.info("Line plot visualized for sales trends across holiday periods.")

    # Histogram for Sales Distribution
    plt.figure(figsize=(14, 7))
    sns.histplot(data=df, x='Sales', hue='HolidayPeriod', bins=30, kde=True, palette="viridis", hue_order=holiday_order)
    plt.title("Sales Distribution by Holiday Period (Histogram)", fontsize=16)
    plt.xlabel("Sales", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    logging.info("Histogram visualized for sales distribution by holiday periods.")

def sales_holiday_summary(df):
    """
    Generate summary statistics for sales across holiday periods.
    Args:
        df (DataFrame): Dataset with 'Sales' and 'HolidayPeriod' columns.
    """
    summary = df.groupby('HolidayPeriod')['Sales'].describe()
    print(summary)
    logging.info("Generated summary statistics for sales behavior across holiday periods.")
    return summary
def add_seasonal_labels(df):
    """
    Add labels for specific seasonal holidays such as Christmas, Easter, etc.
    Args:
        df (DataFrame): Dataset with a 'Date' column.
    Returns:
        DataFrame: Updated DataFrame with a 'Season' column labeling holidays.
    """
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Define the range for Christmas (December 24-26)
    df['Season'] = 'Non-Seasonal'
    df.loc[df['Date'].dt.month == 12, 'Season'] = 'Christmas'

    # Define Easter (assumes Easter dates are pre-defined or calculated)
    # Note: Easter dates vary each year, so you may need to manually include them or calculate dynamically.
    easter_dates = pd.to_datetime(['2022-04-17', '2023-04-09', '2024-03-31'])  # Add more years
    df.loc[df['Date'].isin(easter_dates), 'Season'] = 'Easter'

    # Additional seasons (e.g., New Year, Thanksgiving) can be added in a similar manner

    logging.info("Added seasonal labels (Christmas, Easter) to the dataset.")
    return df
   
def visualize_seasonal_sales_trends(df):
    """
    Visualize the trends of sales during seasonal holidays (e.g., Christmas, Easter) using a line chart.
    Args:
        df (DataFrame): Dataset with 'Sales', 'Season', and 'Date' columns.
    """
    # Ensure that Date is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Group by 'Season' and 'Date' to get the daily average sales per season
    seasonal_sales = df.groupby(['Season', 'Date'])['Sales'].mean().reset_index()

    # Create the line plot for sales trends by season
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=seasonal_sales, x='Date', y='Sales', hue='Season', palette='Set2', linewidth=2)

    # Title and labels
    plt.title("Sales Trends During Seasonal Periods (Christmas, Easter, etc.)", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Average Sales", fontsize=14)
    plt.legend(title="Season", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()

    logging.info("Line chart visualized for sales trends during seasonal periods.")

def analyze_sales_customer_correlation(df):
    """
    Calculate and visualize the correlation between sales and number of customers.
    Args:
        df (DataFrame): Dataset with 'Sales' and 'Customers' columns.
    """
    # Calculate the Pearson correlation coefficient between Sales and Customers
    correlation = df['Sales'].corr(df['Customers'])
    
    # Log the correlation coefficient
    logging.info(f"Pearson correlation between Sales and Customers: {correlation:.2f}")
    
    # Visualize the relationship using a scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Customers', y='Sales', color='b', alpha=0.6)
    
    # Add a regression line for better visualization of the relationship
    sns.regplot(data=df, x='Customers', y='Sales', scatter=False, color='r', line_kws={'linewidth': 2, 'linestyle': '--'})
    
    # Title and labels
    plt.title("Sales vs Number of Customers", fontsize=16)
    plt.xlabel("Number of Customers", fontsize=14)
    plt.ylabel("Sales", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    logging.info("Scatter plot visualized showing the correlation between Sales and Customers.")

    return correlation
