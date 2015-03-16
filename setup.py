
from pip.req import parse_requirements
import os
from setuptools import setup


# https://stackoverflow.com/questions/14399534
filename = os.path.join(os.path.dirname(__file__), 'requirements.txt')
#install_reqs = parse_requirements(filename)
#reqs = [str(ir.req) for ir in install_reqs]
reqs = open(filename).read().splitlines()

setup(
    name='brede',
    version='0.1dev',
    author='Finn Aarup Nielsen',
    author_email='faan@dtu.dk',
    description='Neuroinformatics toolbox',
    license='GPL',
    keywords='neuroinformatics, eeg',
    url='https://github.com/fnielsen/brede',
    py_modules=['brede'],
    install_requires=reqs,
    long_description='',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.2',
        ],
    test_requires=['pytest', 'flake8'],
    )
