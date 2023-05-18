Getting Started
================

Installation
------------

**TMBot** can either be installed using `pip` or directly from the `GitHub <https://github.com/Hayden-Stites/TMBot>`_ repository:

.. code-block:: console

   (env) $ pip install tmbot

or

.. code-block:: console

   (env) $ pip install git+https://github.com/Hayden-Stites/TMBot/src/dist/tmbot-1.0.0.tar.gz

Running The Demo
-----------------

The demo file `demo.py <https://github.com/Hayden-Stites/TMBot/blob/master/demo.py>`_ is available on
the `GitHub <https://github.com/Hayden-Stites/TMBot>`_ repository and shows off basic usage of the **TMBot** package.

To test out your installation, its a good idea to test out a pre-trained model. You can either directly invoke
the `predict_demo` method, or simply run it through the console:

.. code-block:: console

   (env) $ python demo.py -p -m MODEL_PATH

.. note::

   If OpenPlanet is installed somewhere other than the default location, run `demo.py` with the `-o` argument to specify your install location.