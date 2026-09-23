import matplotlib.pyplot
import pandas as pd
import pytest
from matplotlib.figure import Figure

from stimulusbio.visualization import plot_pca


def _pca_scores():
    scores = pd.DataFrame(
        {
            "PC1": [1.0, 2.0, -1.0, -2.0],
            "PC2": [0.5, -0.5, 0.5, -0.5],
        },
        index=pd.Index(
            ["sample_1", "sample_2", "sample_3", "sample_4"], name="sample_id"
        ),
    )
    scores.attrs["explained_variance_ratio"] = (0.6, 0.3)
    return scores


def _metadata():
    return pd.DataFrame(
        {
            "sample_id": ["sample_1", "sample_2", "sample_3", "sample_4"],
            "condition": ["control", "control", "stimulus", "stimulus"],
        }
    )


def test_plot_pca_returns_figure():
    figure = plot_pca(_pca_scores(), _metadata())

    assert isinstance(figure, Figure)


def test_plot_pca_one_scatter_series_per_condition():
    figure = plot_pca(_pca_scores(), _metadata())

    axis = figure.axes[0]

    assert len(axis.collections) == 2


def test_plot_pca_labels_axes_with_explained_variance():
    figure = plot_pca(_pca_scores(), _metadata())

    axis = figure.axes[0]

    assert "60.0%" in axis.get_xlabel()
    assert "30.0%" in axis.get_ylabel()


def test_plot_pca_falls_back_to_plain_labels_without_explained_variance():
    scores = _pca_scores()
    scores.attrs.pop("explained_variance_ratio")

    figure = plot_pca(scores, _metadata())
    axis = figure.axes[0]

    assert axis.get_xlabel() == "PC1"
    assert axis.get_ylabel() == "PC2"


def test_plot_pca_requires_pc1_and_pc2():
    scores = _pca_scores().drop(columns=["PC2"])

    with pytest.raises(ValueError, match="PC1.*PC2"):
        plot_pca(scores, _metadata())


def test_plot_pca_requires_metadata_columns():
    metadata = _metadata().drop(columns=["condition"])

    with pytest.raises(ValueError, match="sample_id.*condition"):
        plot_pca(_pca_scores(), metadata)


def test_plot_pca_rejects_sample_missing_from_metadata():
    metadata = _metadata().iloc[:-1]

    with pytest.raises(ValueError, match="sample_4"):
        plot_pca(_pca_scores(), metadata)


def test_plot_pca_does_not_mutate_inputs():
    scores = _pca_scores()
    metadata = _metadata()
    original_scores = scores.copy(deep=True)
    original_metadata = metadata.copy(deep=True)

    plot_pca(scores, metadata)

    pd.testing.assert_frame_equal(scores, original_scores)
    pd.testing.assert_frame_equal(metadata, original_metadata)


def test_plot_pca_does_not_touch_global_pyplot_state():
    matplotlib.pyplot.close("all")
    assert matplotlib.pyplot.get_fignums() == []

    plot_pca(_pca_scores(), _metadata())

    assert matplotlib.pyplot.get_fignums() == []
