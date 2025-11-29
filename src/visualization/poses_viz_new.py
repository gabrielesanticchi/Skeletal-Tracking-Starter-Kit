"""
Visualization for SMPL poses data.

Provides class-based visualization interface for 3D pose visualization,
pitch tracking, and animation with proper forward kinematics.
"""

import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
