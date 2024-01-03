import os
import csv

def create_csv(folder_path, output_csv):
    with open(output_csv, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['text', 'category'])

        for folder_name in os.listdir(folder_path):
            folder_path_full = os.path.join(folder_path, folder_name)
            
            if os.path.isdir(folder_path_full):
                for filename in os.listdir(folder_path_full):
                    file_path = os.path.join(folder_path_full, filename)
                    
                    if os.path.isfile(file_path):
                        with open(file_path, 'r', encoding='utf-8') as text_file:
                            text = text_file.read()
                            csv_writer.writerow([text, folder_name])

# Specify your folder path and output CSV file
folder_path = '/home/patidarritesh/smartsense/raw_data'
output_csv = 'data_sheet2.csv'

# Call the function to create the CSV
create_csv(folder_path, output_csv)