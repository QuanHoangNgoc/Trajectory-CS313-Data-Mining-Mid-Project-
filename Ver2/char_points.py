import numpy as np
import warnings
import data
from data import *

# warnings.filterwarnings("ignore")


def is_zero_vector(v):
    """
    Checks if a vector is a zero vector.
    """
    return np.allclose(v, np.zeros(v.shape))


def is_nan(value):
    """
    Checks if a value is NaN (Not a Number).
    """
    return np.isnan(value).any()


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)  # divide


def dist(A, B):
    return np.linalg.norm(A - B)


####################################################################
### Angle Distance #################################################
####################################################################
def angle_between(A, B, C, D):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1 = np.array(B) - np.array(A)
    v2 = np.array(D) - np.array(C)
    # compute
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    if is_nan(v1_u) or is_nan(v2_u):
        warnings.warn("NaN value detected in angle!")
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_distance(A, B, C, D):
    angle = angle_between(A, B, C, D)
    scale = dist(A, B)
    try:
        assert angle >= 0 and angle <= np.pi
        assert scale >= 0
        if angle <= np.pi / 2:
            return np.sin(angle) * scale
        return scale
    except Exception as e:
        print(e)
        print(angle, scale)
        assert 0


####################################################################
### Mean Distance #################################################
####################################################################
def perpendicular_distance(p, p1, p2):
    if np.array_equal(p1, p2):
        return np.linalg.norm(p - p1)
    return np.linalg.norm(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)


def mean_distance(A, B, C, D):
    l1 = perpendicular_distance(A, C, D)
    l2 = perpendicular_distance(B, C, D)
    assert l1 >= 0 and l2 >= 0
    if l1 == 0 and l2 == 0:
        return np.array(0)
    return (l1**2 + l2**2) / (l1 + l2)


####################################################################
### Compute cost if we replace and not replace #####################
####################################################################
def define_log2(dist):
    assert dist >= 0
    return dist


WEI = (1, 1)  # !!! adjust alpha for distance


def sum_replace_distance(trajectory, s, t):
    if (trajectory[s] == trajectory[t]).all():
        warnings.warn("Special case is detected in sum_replace_distance!")
        return (0, -1)  #!!! Ensure s and t not in a cpt

    sum = dist(trajectory[s], trajectory[t])  # the cost if replace
    no_cost = 0  # the cost if not replace

    for k in range(s, t):  # [s, t-1]
        d_a = angle_distance(
            trajectory[k], trajectory[k + 1], trajectory[s], trajectory[t]
        )
        d_m = mean_distance(
            trajectory[k], trajectory[k + 1], trajectory[s], trajectory[t]
        )
        sum = WEI[0] * define_log2(d_a) + WEI[1] * define_log2(d_m) + sum
        no_cost = define_log2(dist(trajectory[k], trajectory[k + 1])) + no_cost
    return (sum, no_cost)


def greedy_characteristic_points(trajectory, debug=False):
    # Initialize the characteristic points, add the first point as a characteristic point
    unique = []
    for p in trajectory:
        if len(unique) == 0 or (p != unique[-1]).any():
            unique.append(p)

    if len(unique) <= 2:
        return np.array(unique)
    # assert len(unique) >= 2
    unique = np.array(unique)

    i = 0
    j = i + 1
    end = j  # the last point in a cpt
    cp = [0]
    while j < len(unique):
        tmp = sum_replace_distance(unique, i, j)
        j_is_cp = tmp[0]
        not_choose_j = tmp[1]
        if debug:
            print((i, j), j_is_cp, not_choose_j, end="\n")
        if j_is_cp <= not_choose_j:  # better than
            end = j
            j += 1
        else:  # finish a cpt
            i = end  # (ensure > i avoid inf-loop)
            j = i + 1
            end = j
            cp.append(i)

    if cp[-1] != len(unique) - 1:
        cp.append(len(unique) - 1)  # add the last point as a characteristic point
    return unique[cp]


def get_small_data(data, weights=(1, 1), debug=True) -> list:
    global WEI
    WEI = weights
    print(f"Weights is {WEI}")

    small_data = []
    for traj in TQDM(data):
        cp = greedy_characteristic_points(traj, False)
        small_data.append(cp)
        if debug:
            print(
                len(traj) - len(cp),
                f"{len(cp)}/{len(traj)}",
                round((len(traj) - len(cp)) / len(traj) * 100, 2),
                end="\n",
            )

    WEI = (1, 1)
    return small_data
