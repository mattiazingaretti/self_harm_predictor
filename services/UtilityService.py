import kagglehub
from kagglehub import KaggleDatasetAdapter

import requests
import zipfile
from io import BytesIO
import pandas as pd


def load_kaggle_dataset( dataset_name, file_name, verbose = True):
    """
    Loads a dataset from Kaggle using the KaggleHub library.
    
    Args:
        dataset_name (str): The name of the dataset to load.
        file_name (str): The name of the file within the dataset to load.
    
    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    if dataset_name is None or file_name is None:
        raise ValueError("Dataset name and file name must be provided.")
    
    df_path = kagglehub.dataset_download(dataset_name, path=file_name)
    if verbose:
        print(f"Loaded {len(df_path)} rows from {file_name} in the dataset {dataset_name} and stored at {df_path}")
        
    return pd.read_csv(df_path)


def load_dataset_from_zip(url, verbose = True):
    """
    Loads a dataset from a ZIP file located at the specified URL.
    It is assumed that the URL exposes only one csv file for this implementation
    
    Args:
        url (str): The URL of the ZIP file to load.
    
    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    if url is None:
        raise ValueError("URL must be provided.")
    try:
            response = requests.get(url)
            response.raise_for_status()  # Raise HTTP errors
            
            zip_data = BytesIO(response.content)
            
            with zipfile.ZipFile(zip_data) as zip_file:
                csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
                
                if not csv_files:
                    raise ValueError("No CSV files found in the ZIP archive")
                
                with zip_file.open(csv_files[0]) as csv_file:
                    df = pd.read_csv(csv_file)
            
            if verbose:
                print(f"Loaded {len(df)} rows from {csv_files[0]} in the ZIP archive")
                print(df.head())
            
            return df
    
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid ZIP archive")
    except Exception as e:
        print(f"An error occurred: {e}")
