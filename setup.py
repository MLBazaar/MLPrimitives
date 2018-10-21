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
    'Keras>=2.1.6',
    'featuretools>=0.3.1',
    'lightfm>=1.15',
    'networkx>=2.0',
    'numpy>=1.15.2',
    'pandas>=0.23.4',
    'opencv-python>=3.4.0.12',
    'python-louvain>=0.10',
    'scikit-image>=0.13.1',
    'scikit-learn>=0.20.0',
    'scipy>=1.1.0',
    'tensorflow>=1.11.0',
    'xgboost>=0.72.1',
    'iso639>=0.1.4',
    'langdetect>=1.0.7',
    'nltk>=3.3',
    'urllib3==1.23',    # Otherwise, botocore from featuretools fails
]


tests_require = [
    'mlblocks>=0.2.0',
    'pytest>=3.4.2',
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

    # style check
    'flake8>=3.5.0',
    'isort>=4.3.4',

    # fix style issues
    'autoflake>=1.1',
    'autopep8==1.3.4',    # Keep fixed because of flake8 and pycodestyle

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


json_primitives = glob.glob('mlblocks_primitives/**/*.json', recursive=True)
data_files = defaultdict(list)
for primitive_path in json_primitives:
    parts = primitive_path.split('/')
    dir_path = parts[1:-1]
    install_path = os.path.join('mlblocks_primitives', *dir_path)
    data_files[install_path].append(primitive_path)


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
    data_files = list(data_files.items()),
    description="MLBlocks Primitives",
    entry_points = {
        'console_scripts': [
            'mlprimitives=mlprimitives:_main'
        ],
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
    version='0.1.3',
    zip_safe=False,
)
