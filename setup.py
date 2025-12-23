from setuptools import find_packages, setup
from typing import List

HYPEN_WITH_DOT = "-e ."
def get_requirements(file_path: str) -> List[str]:
    """returns the list of requirements.txt"""

    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', "") for req in requirements]

        if HYPEN_WITH_DOT in requirements:
            requirements.remove(HYPEN_WITH_DOT)

    return requirements


setup(
    name='stock_price_prediction',
    version='0.0.1',
    author='Sukrat Singh',
    author_email="",
    description="Production-grade LSTM-based time series forecasting system",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)