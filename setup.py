from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='Trajectories',
    url='https://github.com/ReadingClouds/Trajectories',
    author='Peter Clark',
    author_email='p.clark@reading.ac.uk',
    contributors='Georgios Efstathiou, Jian-Feng Gu, Todd Jones',
    # Needed to actually package something
    packages=['trajectories', 
             ],
    # Needed for dependencies
    install_requires=['numpy', 'scipy', 'dask', 'xarray'],
    # *strongly* suggested for sharing
    version='0.4.0',
    # The license can be anything you like
    license='MIT',
    description='python code to compute trajectories quantities from MONC output.',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.md').read(),
)