from setuptools import setup

install_requires = [
    'stable-baselines3',
    'numpy',
    'gym',
    'torch',
]

setup(
    name='option_baselines',
    version='0.0.1',
    packages=["option_baselines"],
    url='https://github.com/manuel-delverme/option-baselines/',
    license='',
    author='Manuel Del Verme',
    maintainer='Manuel Del Verme',
    author_email='',
    description='',
    install_requires=install_requires,
)
