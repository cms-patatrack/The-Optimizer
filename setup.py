from setuptools import setup

setup(
    name='optimizer',
    version='0.1.4',
    packages=['optimizer', 'optimizer.mopso'],
	install_requires=['scikit-learn','numpy','matplotlib','pandas', 'numba']
)
