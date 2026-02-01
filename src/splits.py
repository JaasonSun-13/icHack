# src/splits.py
"""Walk-forward expanding-window validation splits."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_walk_forward_splits(
    dates: pd.Series,
    val_years: List[int],
    test_year: int,
) -> List[Dict]:
    """Generate walk-forward validation splits.

    Expanding window: train on all years before val_year, validate on val_year.

    Returns list of dicts:
      { "fold": int, "val_year": int,
        "train_idx": np.array, "val_idx": np.array }
    """
    years = dates.dt.year
    splits = []

    for fold, vy in enumerate(val_years):
        train_mask = years < vy
        val_mask = years == vy
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            logger.warning(f"Skipping fold {fold} (val_year={vy}): insufficient data")
            continue
        splits.append({
            "fold": fold,
            "val_year": vy,
            "train_idx": np.where(train_mask)[0],
            "val_idx": np.where(val_mask)[0],
        })
        logger.info(f"  Fold {fold}: train {train_mask.sum()} rows "
                     f"(< {vy}), val {val_mask.sum()} rows ({vy})")

    # Final model: train on everything before test_year
    final_train = years < test_year
    final_test = years == test_year
    splits.append({
        "fold": -1,
        "val_year": test_year,
        "train_idx": np.where(final_train)[0],
        "val_idx": np.where(final_test)[0],
    })
    logger.info(f"  Final: train {final_train.sum()} rows, "
                f"test {final_test.sum()} rows ({test_year})")

    return splits
