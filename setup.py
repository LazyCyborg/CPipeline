from setuptools import setup, find_packages

setup(
    name='EEGPipeline',
    version='0.1.0',
    description='A package for EEG preprocessing and microstate analysis',
    author='Your Name',
    author_email='alexander.engelmark@gmail.com',
    url='https://github.com/LazyCyborg/EEGPipeline',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.0',
        'pandas>=1.0.0',
        'mne>=1.0.0',
        'mne-bids>=0.10',
        'mne-faster>=0.1',
        'mne-icalabel>=0.4',
        'pycrostates>=0.3',
        'antropy>=0.1.4',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
