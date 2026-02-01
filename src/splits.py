"""Walk-forward expanding-window splits. No shuffling, no future leakage."""
import logging
import numpy as np

logger = logging.getLogger(__name__)


def walk_forward_splits(dates, val_years, test_year):
    """Yield (train_idx, val_idx, year) for each fold.

    Folds per spec:
      Fold 0: Train 2017-2019 → Validate 2020
      Fold 1: Train 2017-2020 → Validate 2021
      ...
      Final:  Train 2017-2024 → Test 2025
    """
    years = dates.dt.year
    splits = []

    for vy in val_years:
        tr = np.where(years < vy)[0]
        va = np.where(years == vy)[0]
        splits.append((tr, va, vy))
        logger.info(f"Fold {len(splits)-1}: train<{vy} ({len(tr)}) → val {vy} ({len(va)})")

    # Final: train on everything < test_year, validate on test_year
    tr = np.where(years < test_year)[0]
    va = np.where(years == test_year)[0]
    splits.append((tr, va, test_year))
    logger.info(f"Final: train<{test_year} ({len(tr)}) → test {test_year} ({len(va)})")

    return splits
