from scripts.compare_with_paper import aggregate, render


def test_comparison_renders_model_and_baseline_metrics() -> None:
    folds = [
        {
            "test_tool": "T10",
            "window_counts": {"test": 4},
            "test": {
                "paper_horizons": {
                    "12": {
                        "mse": 0.5,
                        "mae": 0.4,
                        "accuracy": 0.9,
                        "precision_macro": 0.8,
                    }
                },
                "baselines": {
                    "by_horizon": {
                        "12": {
                            "last_value_mse": 0.7,
                            "persistence_class_accuracy": 0.6,
                        }
                    }
                },
            },
        }
    ]

    table = aggregate(folds)
    output = render(table, "mean_over_folds", "macro")

    assert table["mean_over_folds"]["12"]["mse"] == 0.5
    assert table["pooled_by_windows"]["12"]["last_value_mse"] == 0.7
    assert "0.500" in output
    assert "0.700" in output
