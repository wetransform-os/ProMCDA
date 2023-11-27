import os
import setuptools
from setup_utils import get_requirements

current_dirpath = os.path.dirname(os.path.abspath(__file__))
req_filepath = os.path.join(current_dirpath, 'requirements.txt')

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Eclipse Public License 2.0 (EPL-2.0)",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3.9",
    "Topic :: Scientific/Engineering :: Mathematics",
]

setuptools.setup(
    name="ProMCDA",
    version="1.0.2",
    license='EPL v 2.0',
    author="Flaminia Catalli - wetransform GmbH, Matteo Spada - Zurich University of Applied Sciences",
    author_email='flaminia.catalli@wetransform.to, matteo.spada@zhaw.ch',
    description='A probabilistic Multi Criteria Decision Analysis',
    keywords='MCDA, robustness analysis, Monte Carlo sampling, probabilistic approach',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/wetransform-os/ProMCDA',
    classifiers=classifiers,
    platform=['Linux', 'MacOS', 'Windows'],
    python_requires='>=3.9',
    packages=['mcda'],
    test_suite="tests",
    install_requires=get_requirements(req_filepath)
)
