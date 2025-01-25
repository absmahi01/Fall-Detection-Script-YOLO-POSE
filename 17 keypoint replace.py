import pandas as pd
import os

# Dictionary for keypoint replacements
keypoint_mapping = {
    1: 'Nose',
    2: 'Left Eye',
    3: 'Right Eye',
    4: 'Left Ear',
    5: 'Right Ear',
    6: 'Left Shoulder',
    7: 'Right Shoulder',
    8: 'Left Elbow',
    9: 'Right Elbow',
    10: 'Left Wrist',
    11: 'Right Wrist',
    12: 'Left Hip',
    13: 'Right Hip',
    14: 'Left Knee',
    15: 'Right Knee',
    16: 'Left Ankle',
    17: 'Right Ankle'
}

def replace_keypoints(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Input folder does not exist: {input_folder}")
        return

    # Iterate through all CSV files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            print(f"Processing file: {filename}")  # Debugging line
            file_path = os.path.join(input_folder, filename)

            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

            print(f"Columns in {filename}: {df.columns.tolist()}")  # Debugging line

            # Check if there are at least two columns
            if df.shape[1] > 1:
                # Access the second column (index 1)
                second_column = df.iloc[:, 1]

                # Check the data type of the second column
                print(f"Data in second column before replacement: {second_column.unique()}")  # Debugging line

                # Replace keypoint values
                df.iloc[:, 1] = second_column.replace(keypoint_mapping)

                # Save the modified DataFrame to a new CSV file in the output folder
                output_file_path = os.path.join(output_folder, filename)
                try:
                    df.to_csv(output_file_path, index=False)
                    print(f"Processed {filename} and saved to {output_folder}")
                except Exception as e:
                    print(f"Error saving {filename}: {e}")
            else:
                print(f"Not enough columns in {filename}")  # Debugging line

# Define input and output folders
input_folder =  'F:/Fall Data/NoFall/Stand/nf_mask_s_3/nf_mask_s_3_csv'  # Your output folder path
# Your input folder pat
output_folder = 'F:/Fall Data/NoFall/Stand/nf_mask_s_3/nf_mask_s_3_keypoints_csv'  # Your output folder path

# Run the function
replace_keypoints(input_folder, output_folder)