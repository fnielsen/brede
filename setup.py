
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
    version='0.1.dev0',
    author='Finn Aarup Nielsen',
    author_email='faan@dtu.dk',
    description='Neuroinformatics toolbox',
    license='GPL',
    keywords='neuroinformatics, eeg',
    url='https://github.com/fnielsen/brede',
    packages=['brede', 'brede.api', 
              'brede.core', 'brede.core.test', 
              'brede.data', 'brede.data.examples', 'brede.data.test', 
              'brede.eeg', 'brede.eeg.examples', 'brede.eeg.test', 
              'brede.io', 'brede.qa', 'brede.qa', 'brede.stimuli',
              'brede.surface'],
    package_data={'brede.data': 
                  ['brede_database_data/task_to_cognitive_component.csv', 
                   'words_data/cognitive_words.txt',
                   'words_data/neuroanatomy_words.txt',
                   'words_data/neurodisorder_words.txt',
                   'words_data/neuroimaging_method_words.txt',
                   'words_data/task_to_words.csv'
                   ]},
    install_requires=reqs,
    long_description='',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        ],
    test_requires=['pytest', 'flake8'],
    )
