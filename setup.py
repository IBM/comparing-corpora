#  #
#  *****************************************************************
#  #
#  IBM Confidential
#  #
#  Licensed Materials - Property of IBM
#  #
#  (C) Copyright IBM Corp. 2001, 2021 All Rights Reserved.
#  #
#  The source code for this program is not published or otherwise
#  divested of its trade secrets, irrespective of what has been
#  deposited with the U.S. Copyright Office.
#  #
#  US Government Users Restricted Rights - Use, duplication or
#  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
#  #
#  *****************************************************************
import pathlib
import pkg_resources
from setuptools import setup, find_packages

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

setup(name='compcor',
      version='1.0.0',
      description='Bla bla',
      #long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      keywords='Bla bla',
      url='https://github.com/IBM/comparing-corpora',
      author='Language and Conversation team, Haifa lab, IBM Research AI',
      packages=find_packages('src', exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      package_dir={'': 'src'},
      install_requires=[install_requires],
      scripts=[],
      include_package_data=True,
      python_requires='>=3.8',
      zip_safe=False,
      classifiers=[
          "Programming Language :: Python :: 3",
          "Operating System :: OS Independent"
      ]
      )
