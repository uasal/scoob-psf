from setuptools import setup, find_packages

VERSION = '0.1.0' 
DESCRIPTION = 'Package for running simulations of the SCOOB testbed and interfacing with the testbed'
LONG_DESCRIPTION = 'Package for running simulations of the SCOOB testbed and interfacing with the testbed'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="scoobpsf",
        version=VERSION,
        author="Kian Milani",
        author_email="<kianmilani@arizona.edu>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'Coronagraph Instrument'],
        classifiers= [
            "Development Status :: Alpha-0.1.0",
            "Programming Language :: Python :: 3",
        ]
)
