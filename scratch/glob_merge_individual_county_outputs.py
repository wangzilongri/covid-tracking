import os
import glob
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_csv_file(csv_file_path):
    filename = os.path.basename(csv_file_path)
    fips_code = filename.split('_')[2][5:]
    days_from_start = int(filename.split('_')[3].split('.')[0])

    # Load CSV file
    df = pd.read_csv(csv_file_path)

    # Filter rows
    #df_filtered = df[(df['fips'] == fips_code) & (pd.to_datetime(df['datetime']).dt.date == pd.Timestamp('2020-03-12').date() + pd.DateOffset(days=days_from_start-51))]
    #df_filtered = df[(df['fips'] == fips_code) & (df['days_from_start'] == days_from_start)]
    
    return df

if __name__ == '__main__':
    # Find all CSV files
    csv_files = glob.glob('../data/output/individual_county_backtest_by_cutoff/*/*.csv')

    # Process CSV files in parallel and merge into a single DataFrame
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_csv_file, csv_file) for csv_file in csv_files]

        # Track progress with tqdm
        dfs = []
        for f in tqdm(as_completed(futures), total=len(csv_files)):
            df_filtered = f.result()
            if not df_filtered.empty:
                dfs.append(df_filtered)

        # Concatenate filtered DataFrames into a single DataFrame
        df_merged = pd.concat(dfs, ignore_index=True)
    output_fname = "../data/output/glob_mergd_individual_county_cutoff.csv"
    print("Writing {}".format(output_fname))
    # Write merged DataFrame to CSV file
    df_merged.to_csv(output_fname, index=False)

