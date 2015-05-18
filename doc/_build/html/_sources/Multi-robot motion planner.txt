:mod:`planning_sim` --- Motion planning simulation
====================================================
.. module:: planning_sim

The :mod:`planning_sim` module implements classes to simulate a navigation
scenario consisting of one or more mobile robots that autonomously plan their
motion from an initial state to a final state avoiding static obstacles and
other robots, and respecting kinematic (including nonhonolonomic) constraints.

The motion planner is based on the experimental work developed by Michael Defoort
that seeks a near-optimal solution minimizing the time spend by a robot to
complete its mission.

Obstacles and Boundary
-----------------------

------------------------------------
|
------------------------------------

.. autoclass:: planning_sim.Obstacle
    :members:
    :private-members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

------------------------------------
|
------------------------------------

.. autoclass:: planning_sim.RoundObstacle
    :members:
    :private-members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

------------------------------------
|
------------------------------------

.. autoclass:: planning_sim.PolygonObstacle
    :members:
    :private-members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

------------------------------------
|
------------------------------------

.. autoclass:: planning_sim.Boundary
    :members:
    :private-members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

Robot and Kinematic Model
-------------------------

------------------------------------
|
------------------------------------

.. autoclass:: planning_sim.UnicycleKineModel
    :members:
    :private-members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

------------------------------------
|
------------------------------------

.. autoclass:: planning_sim.Robot
    :members:
    :private-members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

WorldSim
--------

------------------------------------
|
------------------------------------

.. autoclass:: planning_sim.WorldSim
    :members:
    :private-members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

Communication message
---------------------

------------------------------------
|
------------------------------------

.. autoclass:: planning_sim.RobotMsg
    :members:
    :private-members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

