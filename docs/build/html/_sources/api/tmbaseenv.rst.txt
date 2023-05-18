TMBaseEnv
================

This class, as its name suggests, is the main base environment used for Trackmania. It inherits from `gymnasium.Env` and follows the
`Gymnasium <https://gymnasium.farama.org/>`_ API. It can optionally use :doc:`tmgui` for additional supervision and sanity checks.

.. code-block:: python

   from tmbot.core import TMBaseEnv

Methods
------------

.. autofunction:: tmbot.core.base.TMBaseEnv.__init__

.. autofunction:: tmbot.core.base.TMBaseEnv.reward
