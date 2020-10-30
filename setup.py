#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md', encoding='utf-8') as history_file:
    history = history_file.read()

install_requires = [
    'Keras>=2.1.6,<2.4',
    'featuretools>=0.6.1,<0.12',
    'iso639>=0.1.4,<0.2',
    'langdetect>=1.0.7,<2',
    'lightfm>=1.15,<2',
    'mlblocks>=0.3.4,<0.4',
    'networkx>=2.0,<3',
    'nltk>=3.3,<4',
    'numpy>=1.15.2,<1.17',
    'opencv-python>=3.4.0.12,<5',
    'pandas>=0.23.4,<0.25',
    'python-louvain>=0.10,<0.14',
    'scikit-image>=0.13.1,<0.15,!=0.14.3',
    'scikit-learn>=0.20.0,<0.21',
    'scipy>=1.1.0,<2',
    'setuptools>=41.0.0',
    'statsmodels>=0.9.0,<1',
    'tensorflow>=1.11.0,<2',
    'xgboost>=0.72.1,<1',
    'docutils>=0.10,<0.16',    # required by botocore
]


setup_requires = [
    'pytest-runner>=2.11.1',
]

tests_require = [
    'pytest>=3.4.2',
    'pytest-cov>=2.6.0',
    'jupyter>=1.0.0,<2',
    'rundoc>=0.4.3,<0.5',
]

development_requires = [
    # general
    'bumpversion>=0.5.3,<0.6',
    'pip>=9.0.1',
    'watchdog>=0.8.3,<0.11',

    # docs
    'm2r>=0.2.0,<0.3',
    'Sphinx>=1.7.1,<3',
    'sphinx_rtd_theme>=0.2.4,<0.5',
    'ipython>=6.5.0',

    # style check
    'flake8>=3.7.7,<4',
    'isort>=4.3.4,<5',

    # fix style issues
    'autoflake>=1.1,<2',
    'autopep8>=1.4.3,<2',

    # distribute on PyPI
    'twine>=1.10.0,<4',
    'wheel>=0.30.0',

    # Advanced testing
    'coverage>=4.5.1,<6',
    'tox>=2.9.1,<4',

    # Jupyter
    'jupyter>=1.0.0',
    'prompt-toolkit<2.1.0,>=2.0.0'
]

extras_require = {
    'test': tests_require,
    'dev': tests_require + development_requires,
}

setup(
    author='MIT Data To AI Lab',
    author_email='dailabmit@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description='Pipelines and primitives for machine learning and data science.',
    entry_points = {
        'console_scripts': [
            'mlprimitives=mlprimitives.cli:main'
        ],
        'mlblocks': [
            'primitives=mlprimitives:MLBLOCKS_PRIMITIVES',
            'pipelines=mlprimitives:MLBLOCKS_PIPELINES'
        ],
        'mlprimitives': [
            'jsons_path=mlprimitives:MLBLOCKS_PRIMITIVES',
        ]
    },
    extras_require=extras_require,
    include_package_data=True,
    install_requires=install_requires,
    keywords='mlblocks mlprimitives pipelines primitives machine learning data science',
    license='MIT license',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    name='mlprimitives',
    packages=find_packages(include=['mlprimitives', 'mlprimitives.*']),
    python_requires='>=3.6,<3.8',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/HDI-Project/MLPrimitives',
    version='0.2.6.dev0',
    zip_safe=False,
)
