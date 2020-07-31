from setuptools import setup, find_packages

extras = {
    'complete': [
        'matplotlib',
        'numpy',
        'pandas==1.0.4',
        'scikit-learn',
        'scipy',
        'category_encoders'
    ]}

setup(name='vgc_adhd',
      version='0.1',
      description='Ecosystem for design and implementation of vergence-based classifiers',
      url='http://github.com/imanolmorata/vergence_adhd_classifier',
      author='Imanol Morata',
      author_email='imanol.morata@gmail.com',
      license='MIT',
      include_package_data=True,
      python_requires='>=3.6',
      extras_require=extras,
      packages=find_packages(include=['vgc_clf.*']),
      zip_safe=False)
