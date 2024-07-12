from setuptools import setup, find_packages

setup(
    name='optimizer',
    version='0.1.4',
    packages=find_packages(include=['optimizer', 'optimizer.*']),
	install_requires=['scikit-learn','numpy', 'numba', 'stable-baselines3']
)
