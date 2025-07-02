# Changelog
All notable changes to this project will be documented in this file.

### [2.0.0] - 2025-07-01
### Major Changes
- Full refactoring of ProMCDA into a modular Python library.
- Jupyter Notebook examples inserted or updated to reflect new import paths and usage patterns.

### Breaking Changes
- Previous scripts and function calls may not work without adaptation.
- Users should migrate by updating their imports and following the new usage examples provided in the updated notebooks 
  (see `demo` and `training` folders).

### Notes
This release marks a significant step towards maintainability, modularity, and easier future extension of ProMCDA.

### [1.1.0] - 2025-05-09

- **Command-Line Interface (CLI):** a CLI enables users to execute ProMCDA directly from the terminal. 
- **Configuration file support:** implements the ability to utilize a configuration file for ProMCDA, allowing users 
  to define parameters and settings in a structured file. This feature enhances reproducibility and simplifies the setup 
  process for complex analyses.
- **Documentation:** comprehensive documentation for the CLI and configuration file features, including usage examples and best practices.

### Initial Release

- Launched ProMCDA with core functionalities, including:
  - Probabilistic Multi-Criteria Decision Analysis capabilities.
  - Support for various decision criteria and uncertainty modeling.
  - Basic visualization tools for decision analysis results.