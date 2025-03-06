import os
import sys
import json
import logging
import pickle
import numpy as np

# If numba is installed import it and use njit decorator otherwise use a dummy decorator
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def dummy_decorator(func):
            return func
        return dummy_decorator



class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    string_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + string_format + reset,
        logging.INFO: grey + string_format + reset,
        logging.WARNING: yellow + string_format + reset,
        logging.ERROR: red + string_format + reset,
        logging.CRITICAL: bold_red + string_format + reset
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
    Logger.error("Uncaught exception", exc_info=(
        exc_type, exc_value, exc_traceback))


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
            Logger.debug("Creating folder '%s'",folder)
            os.makedirs(folder)
        Logger.debug("Saving to '%s'",full_path)
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
        Logger.debug("Saving to '%s'", full_path)
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(dictionary, f, indent=4)

    @classmethod
    def load_csv(cls, filename):
        full_path = os.path.join(cls.working_dir, filename)
        Logger.debug("Loading from '%s'", full_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The file '{full_path}' does not exist.")
        return np.genfromtxt(full_path, delimiter=',', dtype=float)

    @classmethod
    def load_json(cls, filename):
        full_path = os.path.join(cls.working_dir, filename)
        Logger.debug("Loading from '%s'", full_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The file '{full_path}' does not exist.")
        with open(full_path, encoding='utf-8') as f:
            return json.load(f)

    @classmethod
    def save_pickle(cls, obj, filename):
        if not cls.saving_enabled:
            Logger.debug("Saving is disabled.")
            return
        full_path = os.path.join(cls.working_dir, filename)
        folder = os.path.dirname(full_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        Logger.debug("Saving to '%s'", full_path)
        with open(full_path, 'wb') as f:
            pickle.dump(obj, f)

    @classmethod
    def load_pickle(cls, filename):
        full_path = os.path.join(cls.working_dir, filename)
        Logger.debug("Loading from '%s'", full_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The file '{full_path}' does not exist.")
        with open(full_path, 'rb') as f:
            return pickle.load(f)


@njit
def get_dominated(particles, pareto_lenght):
    dominated_particles = np.full(len(particles), False)
    for i, pi in enumerate(particles):
        for j, pj in enumerate(particles):
            if (i < pareto_lenght and j < pareto_lenght) or i == j:
                continue
            if np.any(pi > pj) and \
                    np.all(pi >= pj):
                dominated_particles[i] = True
                break
    return dominated_particles
