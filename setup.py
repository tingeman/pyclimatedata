'''
Created on Mar 24, 2012

@author: tin@byg.dtu.dk

Run from command line:

python setup.py sdist
python setup.py bdist_wininst

This will generate a distribution zip file and a windows executable installer
Can be installed by running from the unzipped temporary directory:

python setup.py install

Or from development directory, in development mode - will reflect changes made
in the original development directory instantly automatically.

python setup.py develop
'''

from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        zip_safe=False,
        name="pyclimatedata",
        version="0.1",
        description="Package for reading and processing climate data, particularly from DMI, but can be modified to read files from any source.",
        author="Thomas Ingeman-Nielsen",
        author_email="thin@dtu.dk",
        url="http://???/",
        keywords=["Weather data", "Climate data", "DMI"],
        classifiers=[
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.9",
            "Development Status :: 3 - Alpha",
            "Operating System :: Microsoft :: Windows",
            "License :: OSI Approved :: GNU General Public License (GPL)",
            "Intended Audience :: Science/Research",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Topic :: Scientific/Engineering", ],
        packages=find_packages(),
        include_package_data=True,
        package_data={
            # If any package contains *.txt files, include them:
            '': ['*.xml', '*.txt', '*.FOR', '*.for', '*.pyf', '*.pyd']},
        long_description="""\
pyclimatedata
----------------
Package for reading and combining data from weather stations.
The package is particularly tailored to DMI provided data files from 
weather stations in Greenland. 
But custom reader classes can be implemented for any consistent source 
of weather data.

""")
