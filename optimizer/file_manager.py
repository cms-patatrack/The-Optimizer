import os
import numpy as np
import json


class FileManager:
    saving_enabled = True
    loading_enabled = False
    working_dir = "tmp"

    @classmethod
    def save_csv(cls, csv_list, filename="file.csv"):
        if not cls.saving_enabled:
            return
        full_path = os.path.join(cls.working_dir, filename)
        folder = os.path.dirname(full_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.savetxt(full_path,
                   csv_list,
                   fmt='%.18f',
                   delimiter=',')

    @classmethod
    def save_json(cls, dictionary, filename="file.json"):
        if not cls.saving_enabled:
            return
        full_path = os.path.join(cls.working_dir, filename)
        folder = os.path.dirname(full_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(full_path, 'w') as f:
            json.dump(dictionary, f, indent=4)

    @classmethod
    def load_csv(cls, filename):
        full_path = os.path.join(cls.working_dir, filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The file '{full_path}' does not exist.")
        return np.genfromtxt(full_path, delimiter=',', dtype=float)

    @classmethod
    def load_json(cls, filename):
        full_path = os.path.join(cls.working_dir, filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The file '{full_path}' does not exist.")
        with open(full_path) as f:
            return json.load(f)
