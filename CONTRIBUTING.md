<div align="left">
<img src="https://raw.githubusercontent.com/wetransform-os/ProMCDA/release/logo/ProMCDA_logo.png">
<div>

## Contributing to Probabilistic Multi Criteria Decision Analysis, ```ProMCDA```.

### Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Getting started](#getting-started)
- [Issue tracking](#issue-tracking)
- [Making changes](#making-changes)
- [Code style](#code-style)
- [Testing](#testing)
- [Project origins and author affiliations](#affiliation)
- [Mantainers](#mantainers)

### Introduction
Welcome to ```ProMCDA```! We're thrilled that you're interested in contributing to our project. Whether you're fixing 
a bug, implementing a new feature, or improving documentation, your contributions help make ```ProMCDA``` better for 
everyone.

Before you get started, please take a moment to review the following guidelines to ensure that your contributions are 
effective and in line with our project's standards. If you have any questions or need assistance, don't hesitate to 
reach out to the maintainers.

### Prerequisites
Before contributing to ```ProMCDA```, ensure that you have the following prerequisites installed:

- Python (version Python 3.9)
- Pip (version 23.2.1)
- Other project-specific dependencies listed in the `requirements.txt` file

You can install the Python dependencies listed in the `requirements.txt` file using pip. For example:

```bash
pip install -r requirements.txt
```

### Getting Started
First, you'll need to clone the project repository to your local machine. You can do this by running the following 
command in your terminal:

```bash
git clone https://github.com/wetransform-os/ProMCDA
```

Then, navigate to the project directory and install the required dependencies using pip as described above in 
[Prerequisites](#prerequisites).

If you plan to contribute code, it's recommended to set up a virtual environment for development, for example with 
[conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment):

```bash
conda create --name <my-env>
source activate <my-env>
```

Now you're ready to make changes to the codebase! Feel free to explore the project and make improvements. Before 
submitting a pull request, make sure to test your changes thoroughly. Run any relevant tests and ensure that the code 
meets the project's coding standards, see [Code Style](#code-style). We have used 
[pylint](https://pypi.org/project/pylint/) and [Flake8](https://flake8.pycqa.org/en/latest/) for code quality improvement.

Once you're happy with your changes, push your branch to GitHub and submit a pull request. Be sure to provide a clear 
description of the changes you've made and any relevant information for reviewers.

### Issue tracking
We use GitHub Issues to track bugs, feature requests, and other tasks related to the project. If you encounter any 
issues or have suggestions for improvements, please open a new issue on the 
[GitHub repository](https://github.com/wetransform-os/ProMCDA).

Before opening a new issue, search existing issues to see if the problem or feature request has already been reported.

Navigate to the repository of the project. In the repository menu, click on the "Issues" tab and create a new issue.
Fill out the issue title and description. Be as descriptive as possible, including steps to reproduce the issue 
if applicable. Provide detailed information about the issue, including what you expected to happen and what actually 
happened. If applicable, include code snippets, error messages, or steps to reproduce the issue. 

Optionally, you can assign labels to categorize the issue (e.g., new feature, bug, enhancement, documentation).
We use the following labels to categorize issues:

*Bug*: Indicates a problem with the current implementation that needs to be fixed.
*Feature Request*: Suggests a new feature or enhancement to be added to the project.
*Documentation*: Relates to improvements or issues with project documentation.

Click the green "Submit new issue" button to create the issue when you are ready.

### Making changes
Contributing to the project is easy! Follow these steps to make changes and submit them for review:

#### Fork the repository
Go to the [GitHub repository of the project] (https://github.com/wetransform-os/ProMCDA) and click the "Fork" button 
in the top-right corner to create a copy of the repository in your GitHub account. Clone your forked repository from 
your GitHub account to your local machine using the following command:

```bash
Copy code
git clone https://github.com/your-username/project.git
```

Replace "your-username" with your GitHub username. 

#### Create a branch
Create a new branch to work on your changes using the following command:

```bash
git checkout -b my-feature
```

Replace "my-feature" with a descriptive name for your feature or fix.

#### Make changes
Make the necessary changes to the code, documentation, or other project files.

#### Commit changes
Once you've made your changes, commit them to your local repository using the following commands:

```bash
git add . # or be explicit and add only the changes you really want to push
git commit -m "Add new feature"  # Replace with a descriptive commit message
```

#### Push changes
Push your changes to your GitHub repository:

```bash
git push origin my-feature
```

#### Submit a pull request
Go to your fork of the repository on GitHub (or to the original repository if you did not fork it) and click the 
"New Pull Request" button. Compare the changes you made in your branch with the main branch of the original repository.
Provide a title and description for your pull request, detailing the changes you made. Submit your changes for review.

#### Review and iterate
A project maintainer will review your pull request and provide feedback. Make any requested changes and push them to 
your branch. The pull request will be updated automatically.

#### Merge pull request
Once your pull request is approved, a project maintainer will merge your changes into the main branch of the repository.

### Code style
This code adheres to the [PEP 8 style guide](https://peps.python.org/pep-0008/), which promotes readability and 
consistency in Python code:

- Functions are defined using snake_case.
- Comments start with # and are used to explain code functionality.
- Docstrings (multiline strings enclosed in triple quotes) are used to document non-private functions or methods.
- Indentation is consistent and consists of four spaces per level.
- if `__name__ == "__main__":` block is used to allow the script to be executed directly as well as imported as a module.
- private functions are defined using `_name_of_private_function`.

### Testing
The project uses the unittest framework in Python. We define a function e.g., `add(a, b)` that adds two numbers.
We create a test class `TestAddFunction` that inherits from unittest.TestCase. Inside the test class, we define 
multiple test methods, each testing a specific scenario of the add function. Each test method starts with the word test 
to indicate that it's a test case. Inside each test method, we call the function being tested with specific inputs 
and use assert methods like `assertEqual` to verify the expected behavior. In each test method we use three conceptual
blocks to help reading and understanding the test: `# Given`; `# When`; and `# Then`. 

### Project origins and author affiliations
```ProMCDA``` was initiated by Flaminia Catalli while working at [WEtransform GmbH](https://wetransform.to) and 
Matteo Spada while working at [Zurich University of Applied Sciences (ZHAW)](https://www.zhaw.ch/en/university/). 
Flaminia Catalli was supported by the Future Forest II project funded by the Bundesministerium für Umwelt, 
Naturschutz, nukleare Sicherheit und Verbraucherschutz (Germany), grant Nr. 67KI21002A.
The project has since evolved with contributions from various individuals and organizations. We would like to express 
our gratitude to the reviewers who provided valuable feedback during the review phase of the paper submitted to the 
Journal of Open Software (JOSS). Thank you to Jan Bernoth, Mengbing Li, and Paul Rougieux. Their insights and 
suggestions have significantly contributed to the initial improvement and evolution of ```ProMCDA```.

### Maintainers
If you have any questions, concerns, or suggestions regarding ```ProMCDA```, you can reach out to the project 
maintainers via the following methods:

- **Email:** flaminia.catalli@gmail.com, spdmatteo@gmail.com
- **GitHub Issues:** [Create a new issue](https://github.com/wetransform-os/ProMCDA) on our GitHub repository.

We appreciate your feedback and are here to help with any issues or inquiries you may have.
