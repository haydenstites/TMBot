Callbacks
================

Training callbacks extending from `Stable Baselines 3 <https://github.com/DLR-RM/stable-baselines3>`_ callbacks. Using all callbacks is highly recommended.

Methods
------------
.. code-block:: python

   from tmbot.extras.callbacks import TMPauseOnUpdate

.. autofunction:: tmbot.extras.callbacks.TMPauseOnUpdate.__init__

|

.. code-block:: python

   from tmbot.extras.callbacks import TMSaveOnEpochs

.. autofunction:: tmbot.extras.callbacks.TMSaveOnEpochs.__init__

|

.. code-block:: python

   from tmbot.extras.callbacks import TMResetOnEpoch

.. autofunction:: tmbot.extras.callbacks.TMResetOnEpoch.__init__

|

.. code-block:: python

   from tmbot.extras.callbacks import default_callbacks

.. autofunction:: tmbot.extras.callbacks.default_callbacks

