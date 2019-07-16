from setuptools import setup
setup(
  name = 'pylimn',
  packages = ['pylimn'], # this must be the same as the name above
  version = '1.0.2',
  description = 'pylimn',
  author = 'Deen Freelon',
  author_email = 'dfreelon@gmail.com',
  url = 'https://github.com/dfreelon/pylimn/', # use the URL to the github repo
  download_url = 'https://github.com/dfreelon/pylimn/', 
  install_requires = ['geostring','nltk'],
  keywords = ['natural language processing', 'nlp', 'named entity extraction', 'keyword in context','kwic','stemming'], # arbitrary keywords
  classifiers = [],
  include_package_data=True
)