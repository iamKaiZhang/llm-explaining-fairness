import numpy as np
import pandas as pd
import pytest

from explainrec.data import Dataset


@pytest.fixture
def tiny_dataset() -> Dataset:
    """3 users x 4 items; item 3 has a single rating -> cold (threshold 1)."""
    ratings = pd.DataFrame({
        "user":   [0, 0, 0, 1, 1, 1, 2, 2, 2, 0],
        "item":   [0, 1, 2, 0, 1, 2, 0, 1, 2, 3],
        "rating": [5, 3, 1, 4, 4, 2, 1, 3, 5, 2],
    })
    users = pd.DataFrame(
        {"age": [25, 30, 40], "gender": ["M", "F", "M"], "occupation": ["a", "b", "c"]},
    )
    users.index.name = "user"
    items = pd.DataFrame({
        "title": ["A", "B", "C", "D"],
        "genres": [["Drama"], ["Comedy"], ["Drama", "Comedy"], ["Horror"]],
    })
    items.index.name = "item"
    return Dataset(ratings=ratings, users=users, items=items, cold_threshold=1)


@pytest.fixture
def tiny_r_hat() -> np.ndarray:
    # rows: users, cols: items; deliberately distinct optima
    return np.array([
        [5.0, 4.0, 3.0, 1.0],
        [4.0, 5.0, 3.0, 1.0],
        [3.0, 4.0, 5.0, 1.0],
    ])
