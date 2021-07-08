from setuptools import setup, find_packages

extras = {
    'complete': [
        'numpy',
        'pandas==1.3.0',
        'scikit-learn',
        'scipy',
        'category_encoders'
    ]}

setup(name='bio-signal-binary-classifier',
      version='0.1',
      description='Ecosystem for design and implementation of bio-signals-based binary classifiers',
      url='http://github.com/imanolmorata/biosignal_classifier',
      author='Imanol Morata',
      author_email='imanol.morata@gmail.com',
      license='unknown',
      include_package_data=True,
      python_requires='>=3.6',
      extras_require=extras,
      packages=find_packages(include=['b2s_clf.*']),
      zip_safe=False)
