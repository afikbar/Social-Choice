from itertools import combinations
from sklearn.decomposition import PCA
from typing import Callable, Optional
import numpy as np
import csv


def load_profiles(path: str) -> np.ndarray:
    """Reads a csv of voters preferences and return a NumPy array representing it.

    Args:
        path (str): path to csv file containing voters preferences.

    Returns:
        np.ndarray: NumPy array of profiles.
    """
    profiles = np.loadtxt(path, delimiter=',', dtype=int)
    return profiles


def _random_argsort(a: np.ndarray, desc=False) -> np.ndarray:
    """Sorting array 'a' and return the sorted indices.
    Tie-breaking is randomized.

    Args:
        a (np.ndarray): Input array to be sorted.
        desc (bool, optional): Sorting order, if True the sort will be descendingly. Defaults to False.

    Returns:
        np.ndarray: NumPy array of sorted indices.
    """
    rnd_arr = np.random.rand(a.size)
    return np.lexsort((rnd_arr, (-1 * desc) * a))


def _positional_scoring(profiles: np.ndarray, score_vec, weights: np.ndarray):
    positions = np.argsort(profiles)
    scored_positions = np.vectorize(score_vec.__getitem__)(positions)
    return weights.dot(scored_positions)  # Column-wise weighted average


def _borda_score(profiles: np.ndarray, weights: Optional[np.ndarray] = None):
    n, m = profiles.shape
    score_vec = np.arange(m - 1, -1, step=-1)
    weights = np.ones(n) if weights is None else weights
    scores = _positional_scoring(profiles, score_vec, weights)
    return scores


def borda(profiles: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculates the Borda score and returns the candidates ranking.
    Tie-breaking is randomized.

    Args:
        profiles (np.ndarray): Voters profiles, (Voters, Candidates).
        weights (Optional[np.ndarray], optional): Voters weights. Defaults to None.

    Returns:
        np.ndarray: Ranking of candidates over scores.
    """
    scores = _borda_score(profiles, weights)
    # print(f"Scores: {list(scores)}")
    return _random_argsort(scores, desc=True)


def _pairwise_elections(sorted_indices: np.ndarray, c1: int, c2: int, weights: np.ndarray):
    # Strict winning only:
    c1_score = (sorted_indices[:, c1] < sorted_indices[:, c2]).dot(weights)
    c2_score = (sorted_indices[:, c1] > sorted_indices[:, c2]).dot(weights)

    if c1_score > c2_score:
        return c1
    if c2_score > c1_score:
        return c2

    return np.random.choice([c1, c2])


def _copeland_score(profiles: np.ndarray, weights: Optional[np.ndarray] = None):
    n, m = profiles.shape
    scores = np.zeros(m)
    sorted_indices = np.argsort(profiles)  # Sort once
    weights = np.ones(n) if weights is None else weights
    for c1, c2 in combinations(range(m), 2):
        winner = _pairwise_elections(sorted_indices, c1, c2, weights)
        scores[winner] += 1
    return scores


def copeland(profiles: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculates the Copeland score and returns the candidates ranking.
    Tie-breaking is randmoized.

    Args:
        profiles (np.ndarray): Voters profiles, (Voters, Candidates).
        weights (Optional[np.ndarray], optional): Voters weights. Defaults to None.

    Returns:
        np.ndarray: Ranking of candidates over scores
    """
    scores = _copeland_score(profiles, weights)
    # print(f"Scores: {list(scores)}")
    return _random_argsort(scores, desc=True)


def kt_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates the normalized kendall tau distance of two sorted indices arrays.
    Sorted indices of array x is obtain through: `np.argsort(x)`.
    Args:
        a (np.ndarray): Sorted indices of the first array.
        b (np.ndarray): Sorted indices of the second array.

    Returns:
        float: Kendall-Tau distance.
    """
    m = len(a)
    i, j = np.meshgrid(np.arange(m), np.arange(m))
    ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]),
                                np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
    return ndisordered / (m * (m - 1))  # No need to divide by two


def calculate_proxy_dist(profiles: np.ndarray, metric: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    """Calculates the proxy distance using custom distance metric.

    Args:
        profiles (np.ndarray): Voters profiles, (Voters, Candidates).
        metric (Callable[[np.ndarray, np.ndarray], float]): Distance metric to use.

    Returns:
        np.ndarray: Proxy distance for each voter.
    """
    n, m = profiles.shape
    proxy_dist = np.zeros(n)
    sorted_indices = np.argsort(profiles)  # Sort once
    for ((i, a), (j, b)) in combinations(enumerate(sorted_indices), 2):
        dist = metric(a, b)
        proxy_dist[[i, j]] += dist
    proxy_dist = proxy_dist / (n - 1)
    return proxy_dist


def _estimate_error(voters_pi: np.ndarray):
    return voters_pi


def _set_weights(errors: np.ndarray):
    return 0.5 - errors


def proxy_truth_discovery(profiles: np.ndarray,
                          g: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]) -> np.ndarray:
    """Estimates voters weight using P-TD method and uses voting rule `g` to calculate rankings.

    Args:
        profiles (np.ndarray): Voters profiles, (Voters, Candidates).
        g (Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]): Voting rule

    Returns:
        np.ndarray: Ranking of candidates over scores.
    """
    proxy_dist = calculate_proxy_dist(profiles, kt_distance)
    errors = _estimate_error(proxy_dist)
    weights = _set_weights(errors)
    return g(profiles, weights)


def distance_truth_discovery(profiles: np.ndarray,
                             g: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]) -> np.ndarray:
    """Estimates voters weight using D-TD method and uses voting rule `g` to calculate rankings.

    Args:
        profiles (np.ndarray): Voters profiles, (Voters, Candidates).
        g (Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]): Voting rule

    Returns:
        np.ndarray: Ranking of candidates over scores.
    """
    y = g(profiles)
    y_indices = np.argsort(y)
    sorted_indices = np.argsort(profiles)  # Sort once
    errors = np.apply_along_axis(
        lambda a: kt_distance(a, y_indices), 1, sorted_indices)
    weights = _set_weights(errors)
    return g(profiles, weights)


def _pca_eval(data: np.ndarray, k: int, metric: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    """Evaluates PCA data re-construction, using 'metric'.

    Args:
        data (np.ndarray): Data to reduce.
        k (int): Number of components to reduce to.
        metric (Callable[[np.ndarray, np.ndarray], float]): Distance metric to use.

    Returns:
        np.ndarray: 'metric' score for each record.
    """
    # Fit PCA:
    pca = PCA(k)
    pca_rankings = pca.fit_transform(data)
    # Expand:
    inv_pca_rankings = pca.inverse_transform(pca_rankings)
    distances = [metric(a, b) for a, b in zip(data, inv_pca_rankings)]
    return np.array(distances)

from sklearn.metrics import mean_squared_error
def pca_truth_discovery(profiles: np.ndarray, g: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray], k=5) -> np.ndarray:
    """Estimates voters weight using PCA method and uses voting rule `g` to calculate rankings.

    Args:
        profiles (np.ndarray):  Voters profiles, (Voters, Candidates).
        g (Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]): Voting rule.
        k (int, optional): Number of PCA components. Defaults to 5.

    Returns:
        np.ndarray: Ranking of candidates over scores.
    """
    cand_rankings = np.argsort(profiles)  # Columns are "candidates"
    errors = _pca_eval(cand_rankings, k, kt_distance)
    weights = _set_weights(errors)
    return g(profiles, weights)


def main():
    print("Loading 'votes.csv':")
    profiles = load_profiles('votes.csv')
    print("Writing 'estimations.csv'...")
    with open("estimations.csv", 'w', newline='') as file:
        wr = csv.writer(file)
        wr.writerow(distance_truth_discovery(profiles, borda))
        wr.writerow(distance_truth_discovery(profiles, copeland))
        wr.writerow(proxy_truth_discovery(profiles, borda))
        wr.writerow(proxy_truth_discovery(profiles, copeland))
        wr.writerow(borda(profiles))
        wr.writerow(copeland(profiles))
        wr.writerow(pca_truth_discovery(profiles, borda))
        wr.writerow(pca_truth_discovery(profiles, copeland))
    print("Done.")


if __name__ == '__main__':
    main()
