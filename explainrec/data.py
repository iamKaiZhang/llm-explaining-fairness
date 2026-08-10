"""MovieLens 100k loading and dataset context.

Downloads the dataset on first use into ``<repo>/data/`` and exposes a
``Dataset`` object that the rest of the pipeline treats as read-only
context: ratings, user attributes, item metadata, and derived facts
such as cold items.

MovieLens 100k ids are contiguous (users 1..943, items 1..1682), so we
use ``id - 1`` as the 0-based matrix index throughout.
"""

from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


@dataclass
class Dataset:
    """Read-only context shared by the estimator, optimizer, and LLM layer."""

    ratings: pd.DataFrame  # columns: user, item (0-based), rating
    users: pd.DataFrame    # index: user (0-based); columns: age, gender, occupation
    items: pd.DataFrame    # index: item (0-based); columns: title, genres (list[str])
    cold_threshold: int = 20
    n_users: int = field(init=False)
    n_items: int = field(init=False)
    item_counts: np.ndarray = field(init=False)   # ratings per item
    cold_items: np.ndarray = field(init=False)    # 0-based indices

    def __post_init__(self) -> None:
        self.n_users = int(self.ratings["user"].max()) + 1
        self.n_items = int(self.ratings["item"].max()) + 1
        counts = np.zeros(self.n_items, dtype=int)
        np.add.at(counts, self.ratings["item"].to_numpy(), 1)
        self.item_counts = counts
        self.cold_items = np.flatnonzero(counts <= self.cold_threshold)

    def items_with_genre(self, genre: str) -> np.ndarray:
        matches = [g.lower() for g in GENRES]
        if genre.lower() not in matches:
            raise ValueError(f"unknown genre {genre!r}; known: {GENRES}")
        canonical = GENRES[matches.index(genre.lower())]
        mask = self.items["genres"].apply(lambda gs: canonical in gs)
        return np.flatnonzero(mask.to_numpy())

    def popular_items(self, top: int) -> np.ndarray:
        return np.argsort(-self.item_counts)[:top]

    def title(self, item: int) -> str:
        return str(self.items.loc[item, "title"])

    def summary(self) -> str:
        """Compact dataset facts for the LLM system prompt."""
        return (
            f"{self.n_users} users, {self.n_items} movies, "
            f"{len(self.ratings)} ratings (1-5 stars). "
            f"Cold items: the {len(self.cold_items)} movies with at most "
            f"{self.cold_threshold} ratings. "
            f"User attributes: age, gender (M/F), occupation. "
            f"Genres: {', '.join(GENRES)}."
        )


def download_ml100k(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Download and extract MovieLens 100k; return the ml-100k directory."""
    ml_dir = data_dir / "ml-100k"
    if (ml_dir / "u.data").exists():
        return ml_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ml-100k.zip"
    if not zip_path.exists():
        print(f"Downloading MovieLens 100k to {zip_path} ...")
        urllib.request.urlretrieve(ML100K_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    return ml_dir


def load_dataset(data_dir: Path = DEFAULT_DATA_DIR, cold_threshold: int = 20) -> Dataset:
    ml_dir = download_ml100k(data_dir)

    ratings = pd.read_csv(
        ml_dir / "u.data", sep="\t", names=["user", "item", "rating", "ts"],
    )[["user", "item", "rating"]]
    ratings["user"] -= 1
    ratings["item"] -= 1

    users = pd.read_csv(
        ml_dir / "u.user", sep="|",
        names=["user", "age", "gender", "occupation", "zip"],
    ).set_index("user")
    users.index -= 1
    users = users[["age", "gender", "occupation"]]

    item_cols = ["item", "title", "release", "video_release", "url", *GENRES]
    items = pd.read_csv(
        ml_dir / "u.item", sep="|", names=item_cols, encoding="latin-1",
    ).set_index("item")
    items.index -= 1
    genre_matrix = items[GENRES].to_numpy(dtype=bool)
    items = items[["title"]].copy()
    items["genres"] = [
        [g for g, flag in zip(GENRES, row) if flag] for row in genre_matrix
    ]

    return Dataset(ratings=ratings, users=users, items=items, cold_threshold=cold_threshold)
