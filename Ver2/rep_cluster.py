import data
from data import *


# Slope to angle in degrees
def slope_to_angle(slope, degrees=True):
    """
    Convert slope to angle in degrees.
    """
    if not degrees:
        return np.arctan(slope)
    return np.arctan(slope) * 180 / np.pi


# Slope to rotation matrix
def slope_to_rotation_matrix(slope):
    """
    Convert slope to rotation matrix.
    """
    return np.array([[1, slope], [-slope, 1]])


# Get cluster majority line orientation
def get_average_direction_slope(line_list):
    """
    Get the cluster majority line orientation.
    Returns 1 if the lines are mostly vertical, 0 otherwise.
    """
    # Get the average slopes of all the lines
    slopes = []
    for line in line_list:
        slopes.append(
            (line[-1, 1] - line[0, 1]) / (line[-1, 0] - line[0, 0])
            if (line[-1, 0] - line[0, 0]) != 0
            else 0
        )
    slopes = np.array(slopes)

    # Get the average slope
    return np.mean(slopes)


# Trajectory Smoothing
def smooth_trajectory(trajectory, window_size=5):
    """
    Smooth a trajectory using a moving average filter.
    """
    # Ensure that the trajectory is a numpy array of shape (n, 2)
    if not isinstance(trajectory, np.ndarray):
        raise TypeError("Trajectory must be a numpy array")
    elif trajectory.shape[1] != 2:
        raise ValueError("Trajectory must be a numpy array of shape (n, 2)")

    # Ensure that the window size is an odd integer
    if not isinstance(window_size, int):
        raise TypeError("Window size must be an integer")
    elif window_size % 2 == 0:
        raise ValueError("Window size must be an odd integer")

    # Pad the trajectory with the first and last points
    padded_trajectory = np.zeros((trajectory.shape[0] + (window_size - 1), 2))
    padded_trajectory[window_size // 2 : window_size // 2 + trajectory.shape[0]] = (
        trajectory
    )
    padded_trajectory[: window_size // 2] = trajectory[0]
    padded_trajectory[-window_size // 2 :] = trajectory[-1]

    # Apply the moving average filter
    smoothed_trajectory = np.zeros(trajectory.shape)
    for i in range(trajectory.shape[0]):
        smoothed_trajectory[i] = np.mean(padded_trajectory[i : i + window_size], axis=0)

    return smoothed_trajectory


#!!! NEW rep: code by Phuoc
def get_representative_trajectory_ver2(lines, min_lines=3):
    """
    Get the sweeping line vector average, optimized to O(n log n).
    """
    # Get the average rotation matrix for all the lines
    average_slope = get_average_direction_slope(lines)
    rotation_matrix = slope_to_rotation_matrix(average_slope)

    # Rotate all lines such that they are parallel to the x-axis
    rotated_lines = []
    for line in lines:
        rotated_lines.append(np.matmul(line, rotation_matrix.T))

    # Let starting_and_ending_points be the set of all starting and ending points of the lines
    starting_and_ending_points = []
    sto = {}
    id = 0
    for line in rotated_lines:
        starting_and_ending_points.append(
            (line[0, 0], line[0, 1], line[-1, 0], line[-1, 1], id)
        )
        starting_and_ending_points.append(
            (line[-1, 0], line[-1, 1], line[0, 0], line[0, 1], id)
        )
        id += 1

    # Sort the events by x-coordinate
    starting_and_ending_points.sort(key=lambda x: x[0])

    representative_points = []

    num_p = 0
    sum = 0
    # Sweep line algorithm using event list
    for event in starting_and_ending_points:
        x1, y1, x2, y2, idx = event

        val = sto.get(idx)
        if val == None:
            # Add the line y-coordinate to active lines
            sum += y1 + y2
            num_p += 1
            sto[idx] = True

        # If the number of active lines is greater than or equal to min_lines, calculate the average y
        if num_p >= min_lines:
            y_avg = sum / (num_p * 2)
            representative_points.append(np.array([x1, y_avg]))

        if val == True:
            # Remove the line y-coordinate from active lines
            num_p -= 1
            sum -= y1 + y2

    if len(representative_points) == 0:
        warnings.warn("WARNING: No representative points were found.")
        return np.array([])

    # Undo the rotation for the generated representative points
    representative_points = np.array(representative_points)
    representative_points = np.matmul(
        representative_points, np.linalg.inv(rotation_matrix).T
    )

    return representative_points


def get_rep(lines, min_lines=3) -> np.array:
    rep = get_representative_trajectory_ver2(lines=lines, min_lines=min_lines)
    return np.array(rep)


def visualize_rep_cluster(rep, sub_segments, ratio=0.1) -> None:
    # Visualize the representative trajectory along with the original segments
    plt.figure(figsize=(8, 6))

    # Plot the original segments
    print(f"Sample with {int(len(sub_segments) * ratio)} segments!!!")
    for start, end in random.sample(sub_segments, int(len(sub_segments) * ratio)):
        plt.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            "gray",
            linestyle="--",
            # marker="x"
        )

    # Plot the representative trajectory
    plt.plot(
        rep[:, 0],
        rep[:, 1],
        "r",
        # linestyle="--",
        # marker="x",
        label="Representative Trajectory",
    )

    plt.title("Representative Trajectory of a Cluster")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.show()
