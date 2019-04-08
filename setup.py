from setuptools import setup, find_packages

setup(
    name='provenancefiltering',
    version=open('version.txt').read().rstrip(),
    url='',
    license='',
    author='Notre Dame Team',
    author_email='allansp84@gmail.com',
    description='Provenance Filtering for Multimedia Phylogeny',
    long_description=open('README.md').read(),

    packages=find_packages(where='.'),

    # install_requires=open('requirements.txt').read().splitlines(),

    entry_points={
        'console_scripts': [
            'filtering_icip17.py = provenancefiltering.icip17.scripts.filtering_icip17:main',
        ],
    },

)
