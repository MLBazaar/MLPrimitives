#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os
from collections import defaultdict
from setuptools import setup, find_packages


with open('README.md') as readme_file:
    readme = readme_file.read()


with open('HISTORY.md') as history_file:
    history = history_file.read()


install_requires = [
    'Keras>=2.1.6,<3',
    'featuretools>=0.6.1,<0.7',
    'iso639>=0.1.4,<0.2',
    'langdetect>=1.0.7,<2',
    'lightfm>=1.15,<2',
    'networkx>=2.0,<3',
    'nltk>=3.3,<4',
    'numpy>=1.15.2,<1.17',
    'opencv-python>=3.4.0.12,<5',
    'pandas>=0.23.4,<0.25',
    'python-louvain>=0.10,<0.14',
    'scikit-image>=0.13.1,<0.15',
    'scikit-learn>=0.20.0,<0.21',
    'scipy>=1.1.0,<2',
    'tensorflow>=1.11.0,<2',
    'xgboost>=0.72.1,<1',
]


tests_require = [
    'mlblocks>=0.3.0',
    'pytest>=3.4.2',
    'pytest-cov>=2.6.0',
    'google-compute-engine==2.8.12',    # required by travis
]


setup_requires = [
    'pytest-runner>=2.11.1',
]


development_requires = [
    # general
    'bumpversion>=0.5.3',
    'pip>=9.0.1',
    'watchdog>=0.8.3',

    # docs
    'm2r>=0.2.0',
    'Sphinx>=1.7.1',
    'sphinx_rtd_theme>=0.2.4',
    'recommonmark>=0.4.0',
    'ipython==6.5.0',

    # style check
    'flake8>=3.7.7',
    'isort>=4.3.4',

    # fix style issues
    'autoflake>=1.1',
    'autopep8>=1.4.3',

    # distribute on PyPI
    'twine>=1.10.0',
    'wheel>=0.30.0',

    # Advanced testing
    'coverage>=4.5.1',
    'tox>=2.9.1',
]


extras_require = {
    'test': tests_require,
    'dev': tests_require + development_requires,
}


setup(
    author="MIT Data To AI Lab",
    author_email='dailabmit@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="MLBlocks Primitives",
    entry_points = {
        'console_scripts': [
            'mlprimitives=mlprimitives.cli:main'
        ],
        'mlprimitives': [
            'jsons_path=mlprimitives:MLPRIMITIVES_JSONS_PATH'
        ]
    },
    extras_require=extras_require,
    install_requires=install_requires,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='mlblocks mlprimitives mlblocks_primitives',
    name='mlprimitives',
    packages=find_packages(include=['mlprimitives', 'mlprimitives.*']),
    python_requires='>=3.5',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/HDI-Project/MLPrimitives',
    version='0.1.9',
    zip_safe=False,
)
