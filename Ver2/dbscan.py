from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import data
from data import *
from char_points import *


def extract_feature_vector(traj):
    assert len(traj) >= 2

    vectors = []
    for i in range(1, len(traj)):
        direct = traj[i] - traj[i - 1]
        lnorm = np.linalg.norm(direct)
        if lnorm != 0:
            direct = direct / lnorm
        else:
            direct = np.array([0, 0])

        feat = np.array([traj[i - 1], traj[i], direct]).flatten()  #!!! flatten
        feat = np.append(feat, lnorm)
        vectors.append(feat)
        assert len(feat) == 7

    scaler = StandardScaler()  #!!! Scale features before KNN
    vectors = scaler.fit_transform(vectors)
    return np.array(vectors)


def to_vectors(data) -> np.array:
    vectors = None
    for traj in data:
        if len(traj) < 2:
            continue
        if vectors is None:
            vectors = extract_feature_vector(traj)
        else:
            tmp = extract_feature_vector(traj)
            vectors = np.concatenate((vectors, tmp), axis=0)

    return vectors


def vectors_to_labels(vectors, C, min_samples) -> tuple:
    # Create a DBSCAN object with parameters
    dbscan = DBSCAN(
        eps=C, min_samples=min_samples
    )  # Adjust eps and min_samples as needed
    dbscan.fit(vectors)
    labels = dbscan.labels_
    # Identify noise points (cluster -1)
    noise_indices = np.where(labels == -1)[0]
    return labels, noise_indices


def to_segments(data) -> np.array:
    segments = []
    for traj in data:
        if len(traj) < 2:
            continue
        for i in range(1, len(traj)):
            seg = np.array([traj[i - 1], traj[i]])
            segments.append(seg)
    return np.array(segments)


def get_segments_with_clusid(clusid, segments, labels) -> np.array:
    ids = np.where(labels == clusid)[0].tolist()
    sub_segments = [segments[i] for i in ids]  # Use list comprehension
    return sub_segments


def test_cluster(segments, labels, noise_indices):
    print(f"# of segments: {len(segments)}")
    print(f"# of labels: {len(labels)}")
    print(f"# of unique clusters: {max(labels)}")
    for i in range(max(labels)):
        print(
            f"# of segments in cluster {i}: {len(get_segments_with_clusid(i, segments, labels))}"
        )
    print(f"# of noise indices: {len(noise_indices)}")


INF = np.float64("1000")
from sklearn.neighbors import NearestNeighbors


def calculate_parallel_distance(segment1, segment2):
    # Extract points from each segment
    s_i, e_i = np.array(segment1[0]), np.array(segment1[1])
    s_j, e_j = np.array(segment2[0]), np.array(segment2[1])

    # Calculate direction vector of the first segment
    direction = e_i - s_i
    direction_norm = direction / np.linalg.norm(direction)

    # Project 2 onto the direction of the first segment
    l_parallel_1 = np.dot(s_j - s_i, direction_norm)
    l_parallel_2 = np.dot(e_j - e_i, direction_norm)
    l_parallel_3 = np.dot(s_j - e_i, direction_norm)
    l_parallel_4 = np.dot(e_j - s_i, direction_norm)

    # Calculate d_parallel as the minimum of the two projections
    d_parallel = min(abs(l_parallel_1), abs(l_parallel_2))
    d_parallel = min(d_parallel, l_parallel_3)
    d_parallel = min(d_parallel, l_parallel_4)
    return d_parallel


def compute_my_distance(a, b):
    dm = mean_distance(a[0], a[1], b[0], b[1]) + mean_distance(b[0], b[1], a[0], a[1])
    da = angle_distance(a[0], a[1], b[0], b[1]) + angle_distance(b[0], b[1], a[0], a[1])
    dp = calculate_parallel_distance(a, b) + calculate_parallel_distance(b, a)
    return (dm + da + dp) / 2


def vectors_to_labels_version2(vectors, segments, C, min_samples) -> tuple:
    # Use KD-tree for efficient neighborhood search
    nbrs = NearestNeighbors(n_neighbors=100, algorithm="kd_tree").fit(vectors)
    distances, indices = nbrs.kneighbors(vectors)

    # Create a square distance matrix based on KD-tree neighbors
    num_segments = len(segments)
    dist_matrix = np.full((num_segments, num_segments), INF)

    for i in TQDM(range(num_segments)):
        dist_matrix[i, indices[i]] = compute_my_distance(
            segments[i], segments[indices[i]]
        )
        dist_matrix[indices[i], i] = dist_matrix[i, indices[i]]  # Ensure symmetry

    # Create a DBSCAN object with custom matrix
    dbscan = DBSCAN(eps=C, min_samples=min_samples, metric="precomputed").fit(
        dist_matrix
    )
    labels = dbscan.labels_
    # Identify noise points (cluster -1)
    noise_indices = np.where(labels == -1)[0]
    return labels, noise_indices
