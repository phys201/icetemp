from setuptools import setup

setup(name='icetemp',
      version='1.0',
      description='Package to analyze temperature data from AMANDA/IceCube thermistors',
      url='http://github.com/phys201/icetemp',
      author='amulski, diyaselis, joshvillarreal',
      author_email='alexismulski@g.harvard.edu, ddelgado@g.harvard.edu, joshuavillarreal@college.harvard.edu',
      license='GPLv3',
      packages=['icetemp'],
      install_requires=['numpy', 'pandas'])
