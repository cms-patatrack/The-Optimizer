import os
import sys
import json
import numpy as np
import logging

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

Logger = logging.getLogger("Optimizer")
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
Logger.addHandler(handler)

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    Logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

class Randomizer:
    rng = np.random.default_rng()


class FileManager:
    saving_enabled = True
    loading_enabled = False
    working_dir = "tmp"

    @classmethod
    def save_csv(cls, csv_list, filename="file.csv"):
        if not cls.saving_enabled:
            Logger.debug("Saving is disabled.")
            return
        full_path = os.path.join(cls.working_dir, filename)
        folder = os.path.dirname(full_path)
        if not os.path.exists(folder):
            Logger.debug(f"Creating folder '{folder}'")
            os.makedirs(folder)
        Logger.debug(f"Saving to '{full_path}'")
        np.savetxt(full_path,
                   csv_list,
                   fmt='%.18f',
                   delimiter=',')

    @classmethod
    def save_json(cls, dictionary, filename="file.json"):
        if not cls.saving_enabled:
            Logger.debug("Saving is disabled.")
            return
        full_path = os.path.join(cls.working_dir, filename)
        folder = os.path.dirname(full_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        Logger.debug(f"Saving to '{full_path}'")
        with open(full_path, 'w') as f:
            json.dump(dictionary, f, indent=4)

    @classmethod
    def load_csv(cls, filename):
        full_path = os.path.join(cls.working_dir, filename)
        Logger.debug(f"Loading from '{full_path}'")
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The file '{full_path}' does not exist.")
        return np.genfromtxt(full_path, delimiter=',', dtype=float)

    @classmethod
    def load_json(cls, filename):
        full_path = os.path.join(cls.working_dir, filename)
        Logger.debug(f"Loading from '{full_path}'")
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The file '{full_path}' does not exist.")
        with open(full_path) as f:
            return json.load(f)
