import os
import setuptools
from setup_utils import get_requirements

current_dirpath = os.path.dirname(os.path.abspath(__file__))
req_filepath = os.path.join(current_dirpath, 'requirements.txt')

setuptools.setup(
    name="MCDTool",
    version="0.0.1",
    author="Flaminia Catalli - wetransform GmbH & Matteo Spada - Zurich University of Applied Sciences",
    author_email='fc@wetransform.to, spaa@zhaw.ch',
    description='A probabilistic Multi Criteria Decision Analysis',
    python_requires='>=3.9',
    packages=['mcda', 'tests'],
    install_requires=get_requirements(req_filepath)
)