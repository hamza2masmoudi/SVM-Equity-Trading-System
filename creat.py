import os

def create_project_tree(base_dir):
    """
    Creates the specified project structure under the given base directory.
    Existing directories and files are left untouched.
    """
    structure = {
        "data": ["AAPL_labeled.csv", "AAPL_processed.csv", "AAPL_raw.csv"],
        "models": [],
        "notebooks": [],
        "results": [],
        "src": ["preprocessing.py", "model.py", "model_training.py", "utils.py"],
        "venv": []  # Typically, venv should be created manually.
    }

    # Iterate through the structure and create directories and files
    for folder, files in structure.items():
        folder_path = os.path.join(base_dir, folder)
        
        # Create the directory if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created directory: {folder_path}")
        else:
            print(f"Directory already exists: {folder_path}")
        
        # Create any missing files within the directory
        for file in files:
            file_path = os.path.join(folder_path, file)
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    pass  # Create an empty file
                print(f"Created file: {file_path}")
            else:
                print(f"File already exists: {file_path}")

if __name__ == "__main__":
    # Base directory of your project
    BASE_DIR = os.path.abspath(".")  # Current working directory

    print(f"Setting up project tree in: {BASE_DIR}")
    create_project_tree(BASE_DIR)