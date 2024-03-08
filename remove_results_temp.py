import os

folder_path = r'C:\Users\alexs\PycharmProjects\batch-prompting\scripts\results\tmp\batch_inference-commonsense_qa'

# Get a list of all files in the folder
files = os.listdir(folder_path)

# Iterate over each file in the folder
for file in files:
    # Check if the file has a .json extension
    if file.endswith(".json"):
        # Construct the full path to the file
        file_path = os.path.join(folder_path, file)

        # Remove or delete the file
        os.remove(file_path)
        print(f"Deleted: {file_path}")
