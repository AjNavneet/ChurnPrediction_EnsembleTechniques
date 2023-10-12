import pandas as pd

# Function to read the data file 
def read_data(file_path, **kwargs):
    """
    Read data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.
        **kwargs: Additional keyword arguments for pandas read_csv function.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    raw_data = pd.read_csv(file_path, **kwargs)
    return raw_data

# Function to inspect the data
def inspection(dataframe):
    """
    Perform data inspection on the given DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame to inspect.
    """
    print("Types of the variables we are working with:")
    print(dataframe.dtypes)

    print("Total Samples with missing values:")
    print(dataframe.isnull().any(axis=1).sum())

    print("Total Missing Values per Variable")
    print(dataframe.isnull().sum()

# Function to remove null values
def null_values(df):
    """
    Remove rows with null values from a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to remove null values from.

    Returns:
        pd.DataFrame: The DataFrame with null values removed.
    """
    df = df.dropna()
    return df
