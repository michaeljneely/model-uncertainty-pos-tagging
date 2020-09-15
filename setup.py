import distutils
import io
import os
import subprocess

from setuptools import setup

class PylintCommand(distutils.cmd.Command):
  '''A custom command to run Pylint to check for errors on all source files'''

  # Required to implement 
  user_options = []

  def initialize_options(self):
    pass

  def finalize_options(self):
    pass

  def run(self):
    command = ['pylint']
    command.append(f'--rcfile={os.path.abspath(".pylintrc")}')
    command.append('dl4nlp_pos_tagging')
    self.announce(f'Running command: {command}', level=distutils.log.INFO)
    subprocess.check_call(command)


# Package meta-data
NAME = 'dl4nlp-pos-tagging'
DESCRIPTION = 'University of Amsterdam Deep Learning for Natural Language Processing Fall 2020 Mini Project - POS Tagging'
URL = 'https://github.com/michaeljneely/dl4nlp-pos-tagging'
EMAIL = 'michael.neely@student.uva.nl'
AUTHOR = 'Leila Talha, Michael J. Neely, Stefan F. Schouten'
REQUIRES_PYTHON = '>=3.6'
VERSION = '0.0.1'

REQUIRED = [
  'pandas',
  'nltk',
  'tqdm',
  'numpy',
  'allennlp>=1.0.0',
  'allennlp-models>=1.0.0',
  'scipy',
  'seaborn',
  'gensim',
  'spacy==2.1.9',
  'matplotlib',
  'ipython'
]

SETUP = [
  'pytest-runner',
  'torch',
  'torchtext'
]

TEST = [
  'pylint',
  'pytest',
  'pytest-env',
  'pytest-pylint'
]

EXTRAS = {
  'testing': TEST
}

HERE = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description
try:
  with io.open(os.path.join(HERE, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = '\n' + f.read()
except FileNotFoundError:
  LONG_DESCRIPTION = DESCRIPTION

setup(
  name=NAME,
  version=VERSION,
  description=DESCRIPTION,
  long_description=LONG_DESCRIPTION,
  long_description_content_type='text/markdown',
  author=AUTHOR,
  author_email=EMAIL,
  python_requires=REQUIRES_PYTHON,
  url=URL,
  py_modules=['dl4nlp_pos_tagging'],
  install_requires=REQUIRED,
  setup_requires=SETUP,
  tests_require=TEST,
  extras_require=EXTRAS,
  include_package_data=True,
  license='MIT',
  cmdclass={
    'lint': PylintCommand
  },
  classifiers=[
    # Trove classifiers
    # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
    "Intended Audience :: Science/Research",
    "Development Status :: 1 - Planning",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.6",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
  ]
)
