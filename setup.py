import os
import setuptools
from setup_utils import get_requirements

current_dirpath = os.path.dirname(os.path.abspath(__file__))
req_filepath = os.path.join(current_dirpath, 'requirements.txt')

setuptools.setup(
    name="ProMCDA",
    version="0.0.1",
    author="Flaminia Catalli - wetransform GmbH & Matteo Spada - Zurich University of Applied Sciences",
    author_email='flaminia.catalli@wetransform.to, matteo.spada@zhaw.ch',
    description='A probabilistic Multi Criteria Decision Analysis',
    keywords='MCDA, sensitivity analysis, Monte Carlo sampling, probabilistic approach',
    python_requires='>=3.9',
    packages=['mcda', 'tests'],
    license='EPL v 2.0',
    install_requires=get_requirements(req_filepath)
)