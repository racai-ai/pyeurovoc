from setuptools import setup, find_packages

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setup(
    # $ pip install sampleproject
    name='pyeurovoc',  # Required

    version='1.1.0',  # Required

    description='Python API for multilingual legal document classification with EuroVoc descriptors using BERT models.',  # Required

    long_description=long_description,
    long_description_content_type="text/markdown",
    maintainer="Andrei-Marius Avram",
    maintainer_email="avram.andreimarius@gmail.com",
    author="Andrei-Marius Avram, Vasile Pais, Dan Tufis",
    author_email="avram.andreimarius@gmail.com",

    url='https://github.com/racai-ai/pyeurovoc',  # Optional

    classifiers=[  # Optional
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Human Machine Interfaces',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Filters',
        'Topic :: Text Processing :: General',
        'Topic :: Text Processing :: Indexing',
        'Topic :: Text Processing :: Linguistic',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3'
    ],

    keywords='eurovoc bert legal document classification',  # Optional

    # packages=find_packages(exclude=['jupyter']),  # Required
    packages=find_packages("."),  # Required

    install_requires=['transformers', 'sklearn', 'torch', 'scikit-multilearn', 'pyyaml', 'waitress', 'flask'],  # Optional

    zip_safe=False,
)