from setuptools import setup
setup(
  name = 'meerkat',
  packages = ['meerkat'],
  version = '0.3.7',
  description = 'A program for reciprocal space reconstruction',
  author = 'Arkadiy Simonov, Dmitry Logvinovich',
  author_email = 'aglietto@gmail.com',
  url = 'https://github.com/aglie/meerkat.git',
#  download_url = 
  keywords = ['crystallography', 'single crystal', 'reciprocal space reconstruction'],
  classifiers = [],
  install_requires = ['fabio','h5py','numpy'],
)
