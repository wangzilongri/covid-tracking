import os
import csv
import pickle

from datetime import datetime, timedelta
from multiprocessing import Pool
from tqdm import tqdm


# Define the input and output file paths
input_folder = '../data/output/individual_county_backtest_by_county'
output_file = '../data/output/individual_county_backtest_by_county_compiled.csv'

# Define the start date (March 12, 2020)
start_date = datetime(2020, 3, 12)

# Create a dictionary to store the rows by FIPS and day_from_start combination
rows_by_key = {}

# Define a function to process a CSV file
def process_csv_file(csv_path):
    # Extract the day_from_start from the CSV file name
    day_from_start = int(os.path.basename(csv_path).split('_')[3].split('.')[0])
    # Calculate the corresponding date
    date = start_date + timedelta(days=day_from_start-51)
    # Open the CSV file and loop over the rows
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Check if the row corresponds to the same FIPS and date
            if row['fips'] == os.path.basename(os.path.dirname(csv_path)) and row['datetime'] == str(date):
                # Add the row to the dictionary using the FIPS and day_from_start as the key
                key = os.path.basename(os.path.dirname(csv_path)) + '_' + str(day_from_start)
                if key in rows_by_key:
                    rows_by_key[key].append(row)
                else:
                    rows_by_key[key] = [row]

# Create a function to get a list of CSV files in a directory
def get_csv_files(directory):
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

# Get a list of all CSV files in the input folder
directories = [os.path.join(input_folder, f) for f in os.listdir(input_folder)]
csv_files = []
with Pool() as p:
    with tqdm(total=len(directories), desc='Adding CSV files to list') as pbar:
        for result in p.imap_unordered(get_csv_files, directories):
            csv_files.extend(result)
            pbar.update()

#csv_files = csv_files[:1000]

# Create a pool of worker processes and map the CSV files to the process function
with Pool() as p:
    with tqdm(total=len(csv_files), desc='Processing CSV files') as pbar:
        for i, _ in enumerate(p.imap_unordered(process_csv_file, csv_files)):
            pbar.update()

print("Pickling object")
data_dict = {}
for csv_file in tqdm(csv_files):
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fips = row['fips']
            datetime = row['datetime']
            if fips not in data_dict:
                data_dict[fips] = {}
            if datetime not in data_dict[fips]:
                data_dict[fips][datetime] = []
            data_dict[fips][datetime].append(row)

# save the data_dict to a file
with open('merged_individual_county.p', 'wb') as f:
    pickle.dump(data_dict, f)

# Write the compiled rows to the output file
print("Writing {}".format(output_file))
with open(output_file, 'w', newline='') as csvfile:
    #reader = csv.DictReader(csvfile)
    fieldnames = ['fips', 'datetime'] + reader.fieldnames
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for rows in rows_by_key.values():
        for row in rows:
            writer.writerow(row)

