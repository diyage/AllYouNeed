from typing import *

OBJECT = Tuple[int, Tuple[float, float, float, float]]
# (kind_ind, (x1 y1 x2 y2))
LABEL = List[OBJECT]


POSITION = Tuple[float, float, float, float]
KPS = Tuple[int, POSITION, float]
# (kind_ind, (x1, y1, x2, y2), score)
KPS_VEC = List[KPS]

