# src/policy.py
"""Policy decision tree: translates model outputs to actionable alerts."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

HIGH_RISK_TYPES = {"geopolitics", "macro"}

EVENT_GUIDANCE = {
    "geopolitics": {
        "investor": "Hedge geopolitical risk; consider defence and energy exposure.",
        "firm": "Review supply-chain exposure to affected regions.",
        "entrepreneur": "Delay international expansion; secure existing contracts.",
    },
    "macro": {
        "investor": "Reduce leverage; move to quality / short-duration.",
        "firm": "Stress-test liquidity; draw credit lines pre-emptively.",
        "entrepreneur": "Preserve cash; lock in financing before spreads widen.",
    },
    "crisis": {
        "investor": "Tilt toward defensive sectors; monitor contagion.",
        "firm": "Update continuity plans; review operational resilience.",
        "entrepreneur": "Identify demand shifts early; pivot if needed.",
    },
}


@dataclass
class Decision:
    alert_level: str       # LOW, MEDIUM, HIGH
    risk_prob: float
    social_vol: float
    event_type: str
    confidence: str        # LOW, MEDIUM, HIGH
    drivers: List[str] = field(default_factory=list)
    recommendations: Dict[str, str] = field(default_factory=dict)
    explanation: str = ""


class PolicyTree:
    """Transparent if-else rules over model outputs."""

    def __init__(
        self,
        risk_high: float = 0.7,
        risk_med: float = 0.4,
        vol_high: float = 0.7,
        vol_med: float = 0.4,
    ):
        self.rh = risk_high
        self.rm = risk_med
        self.vh = vol_high
        self.vm = vol_med

    def evaluate(
        self,
        risk_prob: float,
        social_vol: float,
        event_type: str = "other",
        top_drivers: Optional[List[Dict]] = None,
    ) -> Decision:
        top_drivers = top_drivers or []

        # Determine alert level
        if risk_prob >= self.rh or social_vol >= self.vh:
            alert = "HIGH"
        elif risk_prob >= self.rm or social_vol >= self.vm:
            alert = "MEDIUM"
        else:
            alert = "LOW"

        # Boost for high-risk types
        if event_type in HIGH_RISK_TYPES and alert == "MEDIUM":
            alert = "HIGH"

        # Confidence
        if risk_prob >= self.rh and social_vol >= self.vh:
            confidence = "HIGH"
        elif risk_prob >= self.rm and social_vol >= self.vm:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # Drivers
        drivers = [d.get("feature", "unknown") for d in top_drivers[:5]]

        # Recommendations
        recs = EVENT_GUIDANCE.get(event_type, {
            "investor": "Monitor conditions; no immediate action required.",
            "firm": "Continue normal operations with elevated awareness.",
            "entrepreneur": "Stay alert but no changes needed yet.",
        })

        if alert == "LOW":
            recs = {
                "investor": "Business as usual; maintain current positions.",
                "firm": "Normal operations.",
                "entrepreneur": "Proceed with planned initiatives.",
            }

        # Explanation
        parts = [
            f"Alert={alert}  Risk={risk_prob:.0%}  SocialVol={social_vol:.2f}  "
            f"Type={event_type}  Confidence={confidence}",
        ]
        if drivers:
            parts.append(f"Top drivers: {', '.join(drivers)}")

        return Decision(
            alert_level=alert,
            risk_prob=risk_prob,
            social_vol=social_vol,
            event_type=event_type,
            confidence=confidence,
            drivers=drivers,
            recommendations=recs,
            explanation="\n".join(parts),
        )
