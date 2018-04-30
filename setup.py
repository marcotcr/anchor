from setuptools import setup

setup(name='anchor_exp',
      version='0.0.0.5',
      description='Anchor explanations for machine learning models',
      url='http://github.com/marcotcr/anchor',
      author='Marco Tulio Ribeiro',
      author_email='marcotcr@gmail.com',
      license='BSD',
      packages=['anchor'],
      install_requires=[
          'numpy',
          'scipy',
          'spacy',
          'lime',
          'scikit-learn'
      ],
      include_package_data=True,
      zip_safe=False)

