import os
import shutil

def copy_readme_files(src_dir, dest_dir):
    # Iterate over all files and directories in the source directory
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file == "README.md":
                src_path = os.path.join(root, file)
                
                # Construct the relative path within the source directory
                relative_path = os.path.relpath(src_path, src_dir)

                # Construct the destination path
                dest_path = os.path.join(dest_dir, relative_path)

                # Check if the file already exists in the destination
                if not os.path.exists(dest_path):
                    # Create the destination directory if it doesn't exist
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                    # Copy the README.md file to the destination directory
                    shutil.copy(src_path, dest_path)
                    print(f"Copied {file} to {dest_path}")
                else:
                    print(f"{file} already exists in {dest_path}. Skipping.")

# Replace these paths with your actual paths
source_directory = "/Users/abhinay/Documents/GitHub/langchain/templates"
destination_directory = "/Users/abhinay/templets readme files"

copy_readme_files(source_directory, destination_directory)
