from setuptools import setup
setup(
  name = 'meerkat',
  packages = ['meerkat'],
  version = '0.2.3',
  description = 'A program for reciprocal space reconstruction',
  author = 'Arkadiy Simonov, Dmitry Logvinovich',
  author_email = 'arkadiy.simonov@alumni.ethz.ch',
  url = 'https://github.com/aglie/meerkat.git',
#  download_url = 
  keywords = ['crystallography', 'single crystal', 'reciprocal space reconstruction'],
  classifiers = [],
  install_requires = ['fabio','h5py','numpy'],
)
