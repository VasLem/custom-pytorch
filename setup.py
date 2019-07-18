from setuptools import setup


with open('requirements.txt') as inp:
    reqs = inp.readlines()
    install_requires = [req for req in reqs if not req.startswith('http')]
    dependency_links = [req for req in reqs if req.startswith('http')]

setup(name="custom_pytorch", packages=['custom_pytorch'], version='1.0.0',
      long_description='Extension of the pytorch library for customized tools',
      dependency_links=dependency_links, install_requires=install_requires)