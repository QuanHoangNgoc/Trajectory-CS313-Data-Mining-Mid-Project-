BOUND = (39, 41.5, 116, 117.5)
MIN_POINTS = 10

import numpy as np
from tqdm import tqdm as TQDM
import os
import matplotlib.pyplot as plt
import random
import warnings


def in_boundary(coor):  # check if coordinates located in boundary
    if (coor[1] > BOUND[0] and coor[1] < BOUND[1]) and (
        coor[0] > BOUND[2] and coor[0] < BOUND[3]
    ):
        return True
    return False


def preprocessing(trajs, min_points=MIN_POINTS) -> list:
    #!!! ''' delete oulier points and small traj '''
    new_trajs = []
    for i in TQDM(range(len(trajs))):
        curtraj = []
        for j in range(len(trajs[i])):
            if in_boundary(trajs[i][j]) == True:
                curtraj.append(trajs[i][j])
            else:
                if len(curtraj) >= min_points:
                    new_trajs.append(np.array(curtraj))
                curtraj = []

        if len(curtraj) >= min_points:
            new_trajs.append(np.array(curtraj))
            # print(1)

    return new_trajs


def get_data(files, PATH="") -> list:
    data = []
    for file in TQDM(files):
        file = str(file)
        if file.endswith(".txt") == False:
            continue

        file_path = os.path.join(PATH, file)
        try:
            traj = []
            with open(file_path, "r") as f:
                for i, line in enumerate(f):
                    if i == 0:
                        continue  #!!! pass the title
                    content = line.split(",")
                    x, y = float(content[-2]), float(content[-1])
                    traj.append([x, y])
            data.append(np.array(traj))

        except Exception as e:
            print(e)
            print(file)

    print(f"Number of trajectories: {len(data)}")
    data = preprocessing(data)
    print(f"Min points is {MIN_POINTS}")
    print(f"Number of trajectories: {len(data)}")
    return data


def visualize_some_trajs(data, num=5, legend=False) -> list:
    colors = ["r", "g", "b", "c", "m"]
    trajectories = random.sample(data, num)

    plt.figure(figsize=(8, 6))
    for i, traj in enumerate(trajectories):
        plt.plot(
            traj[:, 0],
            traj[:, 1],
            marker="o",
            linestyle="--",
            color=colors[i % len(colors)],
            label=f"Trajectory {i+1}",
        )

    plt.title("Some Trajectories")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    if legend:
        plt.legend()
    plt.show()

    return trajectories


def test_dataset(data):
    i = random.randint(0, len(data) - 1)
    print(f"shape of traj-{i}th:", data[i].shape)
    print("#traj:", len(data))

    number_of_point = 0
    for x in data:
        number_of_point += x.shape[0]
    print("#point:", number_of_point)

    nump = [x.shape[0] for x in data]
    # Create a histogram
    plt.hist(nump, bins=np.arange(min(nump), max(nump) + 1, 1), edgecolor="black")

    # Adding title and labels
    plt.title("Histogram of Data")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    # Show the plot
    plt.show()
