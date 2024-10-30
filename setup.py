from setuptools import setup, find_packages

setup(
    name='CPipeline',
    version='0.1.0',
    description='A package for EEG preprocessing, EEG analysis and NLP analysis of transcribed audio',
    author='Alexander Engelmark',
    author_email='alexander.engelmark@gmail.com',
    url='https://github.com/LazyCyborg/CPipeline',
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    include_package_data=True,  # Include other files specified in MANIFEST.in
    install_requires=[
        'mne>=1.0.0',
        'mne-bids>=0.8.0',
        'mne-faster>=0.1.0',
        'mne-icalabel>=0.1.0',
        'numpy>=1.19.2',
        'pandas>=1.1.5',
        'torch>=1.7.1',
        'torchaudio>=0.7.2',
        'transformers>=4.0.0',
        'textacy>=0.10.0',
        'pycrostates>=0.2.0',
        'antropy>=0.3.0',
        'textacy>=0.10.0',
        'neurone-loader', 
        'torch>=1.7.1',
        'torchaudio>=0.7.2',
        'transformers>=4.0.0',
        'textacy>=0.10.0',
        'pycrostates>=0.2.0',
        'antropy>=0.3.0',
        'textacy>=0.10.0',
        'torch>=1.7.1',
        'torchaudio>=0.7.2',
        'transformers>=4.0.0',
        'textacy>=0.10.0',
        'pycrostates>=0.2.0',
        'antropy>=0.1.6',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'flake8>=3.8.0',
            'black>=20.8b1',
            'isort>=5.7.0',
        ],
        'docs': [
            'sphinx>=3.0',
            'sphinx_rtd_theme>=0.5.0',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',  # Change to Beta or Production/Stable as appropriate
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',  # Change if you use a different license
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            # Example: 'eeg-preproc=eeg_processing.eeg_preproc:main',
            # Add any CLI scripts if you have
        ],
    },
)
