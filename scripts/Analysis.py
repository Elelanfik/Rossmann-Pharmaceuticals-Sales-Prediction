import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import os

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Function to analyze the effect of promos on sales
def analyze_promo_effect(df):
    """
    Analyze the effect of promos on sales, customer count, and existing customers.
    Args:
        df (DataFrame): Dataset with 'Sales', 'Customers', and 'Promo' columns.
    """
    # Classify periods as Promo or No Promo
    df['PromoPeriod'] = df['Promo'].apply(lambda x: 'Promo' if x == 1 else 'No Promo')
    
    # Sales comparison between promo and non-promo periods (using bar plot)
    sales_avg = df.groupby('PromoPeriod')['Sales'].mean()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sales_avg.index, y=sales_avg.values, palette='Set2')
    plt.title("Average Sales Comparison: Promo vs No Promo", fontsize=16)
    plt.xlabel("Promo Period", fontsize=14)
    plt.ylabel("Average Sales", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Compare number of customers during promo and non-promo periods (using bar plot)
    customers_avg = df.groupby('PromoPeriod')['Customers'].mean()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=customers_avg.index, y=customers_avg.values, palette='Set2')
    plt.title("Average Customer Comparison: Promo vs No Promo", fontsize=16)
    plt.xlabel("Promo Period", fontsize=14)
    plt.ylabel("Average Customers", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Check sales per customer during promo periods (using bar plot)
    df['SalesPerCustomer'] = df['Sales'] / df['Customers']
    sales_per_customer_avg = df.groupby('PromoPeriod')['SalesPerCustomer'].mean()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sales_per_customer_avg.index, y=sales_per_customer_avg.values, palette='Set2')
    plt.title("Average Sales Per Customer Comparison: Promo vs No Promo", fontsize=16)
    plt.xlabel("Promo Period", fontsize=14)
    plt.ylabel("Average Sales Per Customer", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    logging.info("Analyzed the effect of promos on sales, customers, and existing customers.")

def analyze_promo_effectiveness(df):
    """
    Analyze the effectiveness of promotions and identify stores that benefit most from promotions.
    Args:
        df (DataFrame): Dataset with 'Store', 'Sales', 'Promo', 'DayOfWeek', 'Open' columns.
    """
    # Filter out stores that are closed (Open == 0)
    df_open = df[df['Open'] == 1]

    # Group by Store and Promo to calculate the average sales during promo and non-promo periods
    promo_sales = df_open.groupby(['Store', 'Promo'])['Sales'].mean().unstack().reset_index()
    promo_sales.columns = ['Store', 'No Promo', 'Promo']

    # Calculate the difference between Promo and No Promo sales for each store
    promo_sales['Sales Difference'] = promo_sales['Promo'] - promo_sales['No Promo']

    # Visualize the impact of promo on sales by store
    plt.figure(figsize=(16, 7))  # Increased figure width for better label visibility
    sns.barplot(data=promo_sales, x='Store', y='Sales Difference', palette='viridis')
    plt.title("Impact of Promotions on Sales by Store", fontsize=16)
    plt.xlabel("Store", fontsize=14)
    plt.ylabel("Sales Difference (Promo - No Promo)", fontsize=14)
    
    # Rotate x-axis labels more (90 degrees) to prevent cutting off
    plt.xticks(rotation=90, ha='center')  
    plt.tight_layout()  # Adjust layout to ensure everything fits
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Identify stores with the most positive sales difference during promotions
    best_stores = promo_sales[promo_sales['Sales Difference'] > 0].sort_values(by='Sales Difference', ascending=False)

    return best_stores


def analyze_promo_effectiveness(df):
    """
    Analyze the effectiveness of promotions and identify stores that benefit most from promotions.
    Args:
        df (DataFrame): Dataset with 'Store', 'Sales', 'Promo', 'DayOfWeek', 'Open' columns.
    """
    # Filter out stores that are closed (Open == 0)
    df_open = df[df['Open'] == 1]

    # Group by Store and Promo to calculate the average sales during promo and non-promo periods
    promo_sales = df_open.groupby(['Store', 'Promo'])['Sales'].mean().unstack().reset_index()
    promo_sales.columns = ['Store', 'No Promo', 'Promo']

    # Calculate the difference between Promo and No Promo sales for each store
    promo_sales['Sales Difference'] = promo_sales['Promo'] - promo_sales['No Promo']

    # Visualize the impact of promo on sales by store
    plt.figure(figsize=(14, 7))
    sns.barplot(data=promo_sales, x='Store', y='Sales Difference', palette='viridis')
    plt.title("Impact of Promotions on Sales by Store", fontsize=16)
    plt.xlabel("Store", fontsize=14)
    plt.ylabel("Sales Difference (Promo - No Promo)", fontsize=14)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels and set alignment to the right
    plt.tight_layout()  # Adjust layout to make sure labels fit
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Identify stores with the most positive sales difference during promotions
    best_stores = promo_sales[promo_sales['Sales Difference'] > 0].sort_values(by='Sales Difference', ascending=False)

    return best_stores


def store_type_promo_effectiveness(df):
    """
    Visualize the effectiveness of promo by StoreType.
    Args:
        df (DataFrame): Dataset to analyze.
    """
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='StoreType', y='Sales', hue='Promo')
    plt.title("Promo Effectiveness by StoreType", fontsize=16)
    plt.xlabel("Store Type", fontsize=14)
    plt.ylabel("Average Sales", fontsize=14)
    plt.xticks(rotation=45)  # Rotate x-axis labels if needed
    plt.tight_layout()  # Adjust layout to ensure labels fit
    plt.show()

    logging.info("Promo effectiveness by StoreType visualized")
def sales_trend_open(df):
    """
    Visualize sales trend by store open/close status.
    Args:
        df (DataFrame): Dataset to analyze.
    """
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='DayOfWeek', y='Sales', hue='Open')
    plt.title("Sales Trend by Day of the Week")
    plt.show()
    logging.info("Sales trend by day of the week visualized")

def weekend_sales_comparison(df):
    """
    Compare weekend sales for stores open all week vs not open all week.
    Args:
        df (DataFrame): Dataset to analyze.
    """
    # Find stores that are open all weekdays (i.e., open every day of the week)
    stores_open_all_weekdays = df[df['Open'] == 1].groupby('Store')['DayOfWeek'].nunique()
    weekday_open_stores = stores_open_all_weekdays[stores_open_all_weekdays == 7].index

    # Calculate average weekend sales (DayOfWeek >= 5)
    weekend_sales = df[df['DayOfWeek'] >= 5].groupby('Store')['Sales'].mean()

    # Separate the weekend sales into stores that are open all week and those that are not
    open_weekend_sales = weekend_sales[weekend_sales.index.isin(weekday_open_stores)].mean()
    non_open_weekend_sales = weekend_sales[~weekend_sales.index.isin(weekday_open_stores)].mean()

    # Prepare data for bar plot
    sales_data = {
        'Store Status': ['Open All Week', 'Not Open All Week'],
        'Average Weekend Sales': [open_weekend_sales, non_open_weekend_sales]
    }
    sales_df = pd.DataFrame(sales_data)

    # Create a bar plot for weekend sales comparison
    plt.figure(figsize=(8, 6))
    sns.barplot(data=sales_df, x='Store Status', y='Average Weekend Sales', palette='coolwarm')
    plt.title("Average Weekend Sales: Stores Open All Week vs Not Open All Week", fontsize=16)
    plt.xlabel("Store Status", fontsize=14)
    plt.ylabel("Average Weekend Sales", fontsize=14)
    plt.tight_layout()  # Adjust layout to make sure labels fit
    plt.show()

    logging.info("Weekend sales comparison visualized")

def assortment_sales_effect(df):
    """
    Visualize sales by assortment type using a pie chart.
    Args:
        df (DataFrame): Dataset to analyze.
    """
    if 'Assortment' in df.columns:
        # Calculate total sales for each assortment type
        assortment_sales = df.groupby('Assortment')['Sales'].sum()

        # Create a pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(assortment_sales, labels=assortment_sales.index, autopct='%1.1f%%', colors=sns.color_palette('Set2', len(assortment_sales)))
        plt.title("Sales Distribution by Assortment Type", fontsize=16)
        plt.ylabel("")  # Remove y-axis label as it's not needed for a pie chart
        plt.tight_layout()  # Adjust layout to make sure everything fits
        plt.show()

        logging.info("Sales distribution by assortment type visualized as a pie chart")
    else:
        logging.warning("Assortment column not found in the dataset.")

        
def competitor_distance_effect(df):
    """
    Visualize the effect of competitor distance on sales.
    Args:
        df (DataFrame): Dataset to analyze.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='CompetitionDistance', y='Sales')
    plt.title("Sales vs Competitor Distance")
    plt.show()
    logging.info("Competitor distance effect on sales visualized")

def sales_competition_trend(df):
    """
    Visualize sales trends with respect to new and old competitors.
    Args:
        df (DataFrame): The dataset.
    """
    df['CompetitionOpened'] = np.where(df['CompetitionDistance'].isna(), 'No Competitor', 'Old Competitor')
    df['CompetitionOpened'] = np.where(df['CompetitionDistance'].notna() & df['CompetitionDistance'].shift(1).isna(), 'New Competitor', df['CompetitionOpened'])

    plt.figure(figsize=(15, 6))
    sns.lineplot(data=df, x='Date', y='Sales', hue='CompetitionOpened')
    plt.title("Sales Trend with New and Old Competitors")
    plt.show()
    logging.info("Sales trend with new and old competitors visualized")

### Correlation Heatmap ###

def correlation_heatmap(df):
    """
    Creates a correlation heatmap of numerical features in the dataset.
    Args:
        df (DataFrame): The dataset.
    """
    df.replace('None', np.nan, inplace=True)  # Replace 'None' with NaN
    df.fillna(0, inplace=True)  # Fill NaN values with 0
    
    # Convert categorical columns into numeric
    label_encoder = LabelEncoder()
    df['StateHoliday'] = label_encoder.fit_transform(df['StateHoliday'].astype(str))
    df['StoreType'] = label_encoder.fit_transform(df['StoreType'].astype(str))
    df['Assortment'] = label_encoder.fit_transform(df['Assortment'].astype(str))

    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    # Generate correlation heatmap
    plt.figure(figsize=(16, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap of Numerical Features')
    plt.show()
    logging.info("Correlation heatmap visualized")