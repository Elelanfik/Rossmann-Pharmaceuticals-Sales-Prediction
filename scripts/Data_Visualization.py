import seaborn as sns
import matplotlib.pyplot as plt

def plot_promo_distribution(data, data_type="Training", logger=None):
    """
    Visualizes the distribution of the 'Promo' column in the dataset (Training or Test).

    Parameters:
        data (DataFrame): The dataset containing the 'Promo' column.
        data_type (str): Indicates if the data is 'Training' or 'Test' data. Default is 'Training'.
        logger (logging.Logger, optional): Logger instance to log information. Default is None.

    Returns:
        None
    """
    # Set the figure size for the plot
    plt.figure(figsize=(8, 6))

    # Create the countplot with custom color palette
    sns.countplot(data=data, x='Promo', palette=['#3498db', '#e74c3c'])  # Use different colors for the two classes

    # Add labels on top of each bar
    for bar in plt.gca().patches:
        plt.gca().text(
            bar.get_x() + bar.get_width() / 2,  # Center of the bar
            bar.get_height() + 10,  # Slightly above the bar
            int(bar.get_height()),  # Display count value
            ha='center', va='center', fontsize=10, color='black'
        )

    # Add title and labels to the plot
    plt.title(f'Promo Distribution in {data_type} Data', fontsize=14, weight='bold')
    plt.xlabel('Promo (0 = No Promo, 1 = Promo)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Horizontal grid lines
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Logging information about the plot
    if logger:
        logger.info(f"Visualized promo distribution in {data_type} data with improved styling.")
