from src.data_loader import get_data

if __name__ == "__main__":
    # Fetch, clean, and save data
    df = get_data()
    print("Data fetching and saving completed.")
    print (df.head())