from __future__ import annotations

import os

import numpy as np
import pandas as pd

from recipes.stnsnr.stats.walk import walk_lfp_fogq_3m_correlation as workflow


def test_load_fogq_uses_established_columns_without_recalculation(tmp_path) -> None:
    input_path = tmp_path / "subj_delta_effect.xlsx"
    source = pd.DataFrame(
        {
            "ID": ["sub-02", "sub-01"],
            workflow.FOGQ_COLUMNS[0]: [25.0, -10.0],
            workflow.FOGQ_COLUMNS[1]: [12.5, 40.0],
            "unused": [1, 2],
        }
    )
    source.to_excel(input_path, index=False)

    actual = workflow.load_fogq_3m_table(input_path)

    expected = source.loc[:, [workflow.PAIR_COL, *workflow.FOGQ_COLUMNS]].sort_values(
        workflow.PAIR_COL
    )
    expected = expected.reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


def test_mounted_source_path_preserves_the_existing_relative_definition() -> None:
    source_path = workflow.LOCAL_BAND_SPECS[0].input_path
    expected = workflow.WALK_DIR / source_path.relative_to(workflow.SOURCE_WALK_DIR)

    assert workflow._mounted_source_path(source_path) == expected
    assert str(expected).startswith(os.path.join(os.environ["MYLFP_PROJECT_ROOT"], 'summary/stats/walk/'))


def test_filter_and_correlation_preserve_pairing_order_and_matrix_contract() -> None:
    x_wide = pd.DataFrame(
        {
            workflow.PAIR_COL: ["sub-04", "sub-02", "sub-01", "sub-03", "sub-extra"],
            "feature_a": [4.0, 2.0, 1.0, 3.0, 8.0],
            "feature_b": [1.0, np.nan, 4.0, 2.0, 9.0],
        }
    )
    fogq = pd.DataFrame(
        {
            workflow.PAIR_COL: ["sub-03", "sub-01", "sub-04", "sub-02"],
            workflow.FOGQ_COLUMNS[0]: [30.0, 10.0, 40.0, 20.0],
            workflow.FOGQ_COLUMNS[1]: [5.0, 20.0, 1.0, 10.0],
        }
    )

    x_common, fogq_common = workflow.filter_to_common_ids(x_wide, fogq)
    result = workflow.compute_walk_fogq_correlation(x_common, fogq_common)

    expected_ids = ["sub-01", "sub-02", "sub-03", "sub-04"]
    assert x_common[workflow.PAIR_COL].tolist() == expected_ids
    assert fogq_common[workflow.PAIR_COL].tolist() == expected_ids
    assert result.r.shape == (2, 2)
    assert result.r.index.tolist() == workflow.FOGQ_COLUMNS
    assert result.r.columns.tolist() == ["feature_a", "feature_b"]
    assert result.n.loc[workflow.FOGQ_COLUMNS[0], "feature_a"] == 4
    assert result.n.loc[workflow.FOGQ_COLUMNS[0], "feature_b"] == 3
    assert result.method == "spearman"
    assert result.correction_mode == "All"
    assert result.correction_method == "bh"
    assert result.p_adj is not None


def test_pivot_feature_rows_keeps_requested_order_and_pairwise_missingness() -> None:
    feature_rows = [
        pd.DataFrame(
            {
                workflow.PAIR_COL: ["sub-01", "sub-02"],
                "feature": ["feature_b", "feature_b"],
                workflow.VALUE_COL: [2.0, 4.0],
            }
        ),
        pd.DataFrame(
            {
                workflow.PAIR_COL: ["sub-01"],
                "feature": ["feature_a"],
                workflow.VALUE_COL: [1.0],
            }
        ),
    ]

    actual = workflow._pivot_feature_rows(
        feature_rows,
        ordered_features=["feature_a", "feature_b"],
        table_name="synthetic walk LFP",
    )

    assert actual.columns.tolist() == [workflow.PAIR_COL, "feature_a", "feature_b"]
    assert actual[workflow.PAIR_COL].tolist() == ["sub-01", "sub-02"]
    assert np.isnan(actual.loc[1, "feature_a"])
    assert actual.loc[1, "feature_b"] == 4.0
