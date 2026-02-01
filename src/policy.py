"""Policy tree: LOW / MEDIUM / HIGH alerts with recommendations."""
import numpy as np


def apply_policy(risk_probs, social_vols, event_types,
                 rh=0.70, rm=0.40, sh=0.70, sm=0.40,
                 boost_types=None):
    """Apply policy tree to arrays of predictions.

    Logic:
      risk >= 0.70 OR svol >= 0.70                → HIGH
      risk >= 0.40 OR svol >= 0.40                → MEDIUM
        + if event_type in {geopolitics, macro}   → upgrade to HIGH
      else                                         → LOW
    """
    if boost_types is None:
        boost_types = ["geopolitics", "macro"]

    alerts = []
    recs = []
    for rp, sv, et in zip(risk_probs, social_vols, event_types):
        if rp >= rh or sv >= sh:
            alert = "HIGH"
        elif rp >= rm or sv >= sm:
            alert = "MEDIUM"
            if et in boost_types:
                alert = "HIGH"
        else:
            alert = "LOW"
        alerts.append(alert)
        recs.append(_recommendation(alert, et))
    return alerts, recs


RECS = {
    ("HIGH", "geopolitics"): "Hedge equity exposure; consider safe-haven assets (gold, Treasuries).",
    ("HIGH", "macro"): "Review fixed-income duration; prepare for rate volatility.",
    ("HIGH", "crisis"): "Increase cash allocation; review tail-risk hedges.",
    ("MEDIUM", "geopolitics"): "Maintain positions but tighten stop-losses.",
    ("MEDIUM", "macro"): "Review upcoming economic calendar; diversify across sectors.",
    ("MEDIUM", "crisis"): "Monitor situation; reduce concentration risk in affected sectors.",
}


def _recommendation(alert, etype):
    if alert == "LOW":
        return "Normal operations. Continue systematic strategy execution."
    return RECS.get((alert, etype),
                    f"{alert} alert — review positions and monitor developments.")


def format_alert(alert):
    icons = {"HIGH": "🔴 HIGH", "MEDIUM": "🟡 MEDIUM", "LOW": "🟢 LOW"}
    return icons.get(alert, alert)
