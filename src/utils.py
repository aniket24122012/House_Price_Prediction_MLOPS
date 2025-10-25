import os

def ensure_folder(file_path):
    folder = os.path.dirname(file_path)
    os.makedirs(folder, exist_ok=True)
    if not os.path.exists(file_path):
        open(file_path, "w").close()
