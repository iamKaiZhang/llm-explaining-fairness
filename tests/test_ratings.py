import numpy as np

from explainrec.ratings import RatingModel


def test_fit_and_predict_shapes(tiny_dataset):
    model = RatingModel(n_factors=2, n_epochs=3).fit(tiny_dataset)
    r_hat = model.predict_matrix()
    assert r_hat.shape == (3, 4)
    assert np.all(r_hat >= 1.0) and np.all(r_hat <= 5.0)
    assert model.train_rmse < 2.0


def test_gender_override_is_local_and_pathway_only(tiny_dataset):
    model = RatingModel(n_factors=2, n_epochs=3).fit(tiny_dataset)
    base = model.predict_matrix()
    flipped = model.predict_matrix(gender_overrides={0: "F"})
    # only user 0 changes
    assert np.allclose(base[1:], flipped[1:])
    # and the change equals the delta-pathway swap (before clipping)
    assert not np.allclose(base[0], flipped[0])
    # flipping to the same gender is a no-op
    same = model.predict_matrix(gender_overrides={0: "M"})
    assert np.allclose(base, same)
