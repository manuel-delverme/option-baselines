import os

from option_baselines.aioc import AIOC
from option_baselines.aoc import AOC
from option_baselines import common
from option_baselines import tabular

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file, "r") as file_handler:
    __version__ = file_handler.read().strip()