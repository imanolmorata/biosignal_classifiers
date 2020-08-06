.. b2s_clf documentation master file, created by
   sphinx-quickstart on Wed Aug  5 11:16:01 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Binary Bio-signal Classifiers (b2s_clf) documentation
================================================================

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   b2s_clf.rst
   b2s_clf.apps.rst
   b2s_clf.dataset_transformer.rst
   b2s_clf.ensemble.rst
   b2s_clf.experiments.rst
   b2s_clf.sampler.rst
   b2s_clf.utils.rst
   modules.rst



Introduction
============

.. automodule:: b2s_clf
    :members:
    :undoc-members:
    :show-inheritance:



============

**b2s_clf** is a small auto-contained machine-learning platform built on top of ``pandas`` and ``scikit-learn`` and
designed to focus in one-dimensional, time-dependant signal data. It was written with bio-signal data in mind, thus it
can handle subject-related data that gives context to the signals put into study (e.g. ECG data that contains patient
information such as age or sex). For that reason, it is designed to work with dual entry data: a ``pandas.DataFrame``
containing the signal data proper and a second ``pandas.DataFrame`` that contains subject data, the only requirement
being that they both share a common subject ID-like column that allows linking the two. This is especially useful when
the signal data contains many signals from the same subject and one wants to perform samplings based on the subject
instead of the signal itself.

The main feature is the class :class:`b2s_clf.ensemble.ensemble_module.Ensemble` that implements a binary ensemble
classifier especially designed to work with both scarce and noisy data. This is a generalization of classical ensemble
models in the sense that it allows any combination of weak classifiers (i.e. it can train trees *and* linear classifiers
at the same time) in the fitting process. The way it re-samples training data to fit those weak classifiers is also
thought to be more general and capable of identifying scarce and specific yet prediction-bearing patterns.

The ensemble fit process or re-sampling the data can actually be customized and performed in a completely separated
fashion through the class :class:`b2s_clf.sampler.sampler_module.Sampler`, designed to be fed with a
``pandas.DataFrame`` and produce as many train and test batches *as one desires*.

Other tools included are the package ``b2s_clf.dataset_transform`` with modules for encoding, normalization and signal
treatment. In particular, the module ``b2s_clf.dataset_transform.compressor`` implements a signal compression algorithm
designed to catch continuous portions of a signal that carry the same relevant information. In next releases, more tools
will be added to this, such as signal smoothing, de-noising and time-alignment.

Finally, the module implements the package ``b2s_clf.apps`` with tools to perform classical ML experiments directly
from the shell. This is achieved thanks to the package ``b2s_clf.experiments``, which implements the super-class
:class:`b2s_clf.experiments.experiment.Experiment` that supports specific experiments through its sub-classes.

Motivation
**********

At least two purposes gave rise to the ``b2s_clf`` project:

* A lack of bio-signal-focused data science and machine-learning libraries in the python ecosystem.
* Exploit the power of ``scikit-learn`` and ``pandas`` to build bio-signal-based classifiers building a platform that
  handles the hard work of sampling and treating signal data that may be stratified in many ways at the same time.

Limitations
***********

- This is a binary classifier library. It does not, nor wants to, implement multi-label classifiers or regressions.
- Currently it only implements ensemble models though it is prepared to support future implementations of other families
  of models such as linear models.
- Still, one can sort of train a linear model by setting an ensemble model whose only
  weak classifier is the linear model itself and the invention will work (we have tested it).
- It is still lacking in performance metrics, at the moment it just implements very basic score computations.
- Future releases intend to offer more load/save options and (hopefully) a better experiment interface.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`