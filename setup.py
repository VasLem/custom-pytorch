import os
from setuptools import setup, find_packages
from setuptools.config import read_configuration
from package import Package, UpdateRequirements
DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG = read_configuration(os.path.join(DIR, 'setup.cfg'))

with open('requirements.txt') as inp:
    reqs = inp.readlines()
    install_requires = [req for req in reqs if not req.startswith('http')]
    dependency_links = [req for req in reqs if req.startswith('http')]

setup(name="custom_pytorch", version=CONFIG['metadata']['version'],
      long_description=CONFIG['metadata']['long_description'],
      dependency_links=dependency_links, install_requires=install_requires,
      packages=find_packages(),
      cmdclass={
          "package": Package,
          "update_reqs": UpdateRequirements})
