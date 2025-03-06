import importlib

# Lazy import to avoid circular dependencies
models = importlib.import_module("promcda.models")
utils = importlib.import_module("promcda.utils")
mcda_functions = importlib.import_module("promcda.mcda_functions")
configuration = importlib.import_module("promcda.configuration")
