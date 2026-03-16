"""Synthetic Player Generator — Probabilistic model for diverse opponent profiles.

Generates realistic synthetic opponent profiles using a probabilistic
graphical model (PGM) approach inspired by NVIDIA's Nemotron-Personas.

Instead of hand-crafting archetypes, this module defines statistical
relationships between player traits and generates diverse, realistic
profiles by sampling from the joint distribution.

Trait dependencies:
    skill_level → {vpip, pfr, aggression, fold_to_cbet, timing_variance}
    tilt_prone → {vpip_delta, aggression_delta, overbet_freq}
    experience → {positional_awareness, sizing_precision, timing_consistency}

Usage::

    generator = SyntheticPlayerGenerator(seed=42)
    profiles = generator.generate_batch(100)
    for p in profiles:
        print(p.name, p.archetype, p.stats)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class SyntheticPlayerStats:
    """Statistical profile of a synthetic player."""

    vpip: float = 0.25
    pfr: float = 0.18
    three_bet: float = 0.06
    fold_to_cbet: float = 0.50
    aggression_factor: float = 2.0
    wtsd: float = 0.28
    cbet_freq: float = 0.65
    check_raise_freq: float = 0.06
    overbet_freq: float = 0.03
    limp_freq: float = 0.05
    squeeze_freq: float = 0.04
    donk_freq: float = 0.08
    timing_mean_ms: float = 2000.0
    timing_std_ms: float = 800.0
    positional_awareness: float = 0.5  # 0=ignores position, 1=expert
    sizing_precision: float = 0.5      # 0=round numbers, 1=precise pot fractions

    def to_vector(self) -> list[float]:
        """Convert to 12-dim vector for StyleEmbedder."""
        return [
            self.vpip, self.pfr, self.three_bet, self.fold_to_cbet,
            self.aggression_factor / 10.0, self.wtsd, self.cbet_freq,
            self.check_raise_freq, self.overbet_freq, self.limp_freq,
            self.squeeze_freq, self.donk_freq,
        ]


@dataclass
class SyntheticPlayer:
    """A generated synthetic player with full profile."""

    name: str
    archetype: str
    skill_level: float         # 0-1
    tilt_propensity: float     # 0-1
    experience_hours: float    # estimated hours played
    stats: SyntheticPlayerStats = field(default_factory=SyntheticPlayerStats)


# Archetype stat templates (mean values)
_ARCHETYPE_TEMPLATES: dict[str, dict[str, float]] = {
    "nit": {
        "vpip": 0.14, "pfr": 0.10, "three_bet": 0.03, "fold_to_cbet": 0.55,
        "aggression_factor": 1.5, "wtsd": 0.22, "cbet_freq": 0.70,
        "check_raise_freq": 0.03, "overbet_freq": 0.01, "limp_freq": 0.02,
        "squeeze_freq": 0.02, "donk_freq": 0.02,
    },
    "tag": {
        "vpip": 0.22, "pfr": 0.18, "three_bet": 0.07, "fold_to_cbet": 0.45,
        "aggression_factor": 2.5, "wtsd": 0.27, "cbet_freq": 0.68,
        "check_raise_freq": 0.07, "overbet_freq": 0.04, "limp_freq": 0.01,
        "squeeze_freq": 0.05, "donk_freq": 0.04,
    },
    "lag": {
        "vpip": 0.30, "pfr": 0.25, "three_bet": 0.10, "fold_to_cbet": 0.38,
        "aggression_factor": 3.5, "wtsd": 0.30, "cbet_freq": 0.72,
        "check_raise_freq": 0.10, "overbet_freq": 0.08, "limp_freq": 0.01,
        "squeeze_freq": 0.08, "donk_freq": 0.05,
    },
    "maniac": {
        "vpip": 0.45, "pfr": 0.35, "three_bet": 0.15, "fold_to_cbet": 0.25,
        "aggression_factor": 4.5, "wtsd": 0.35, "cbet_freq": 0.80,
        "check_raise_freq": 0.12, "overbet_freq": 0.15, "limp_freq": 0.02,
        "squeeze_freq": 0.12, "donk_freq": 0.10,
    },
    "fish": {
        "vpip": 0.50, "pfr": 0.10, "three_bet": 0.02, "fold_to_cbet": 0.60,
        "aggression_factor": 1.0, "wtsd": 0.35, "cbet_freq": 0.40,
        "check_raise_freq": 0.02, "overbet_freq": 0.01, "limp_freq": 0.25,
        "squeeze_freq": 0.01, "donk_freq": 0.15,
    },
    "whale": {
        "vpip": 0.55, "pfr": 0.20, "three_bet": 0.05, "fold_to_cbet": 0.30,
        "aggression_factor": 2.0, "wtsd": 0.40, "cbet_freq": 0.50,
        "check_raise_freq": 0.05, "overbet_freq": 0.10, "limp_freq": 0.15,
        "squeeze_freq": 0.03, "donk_freq": 0.12,
    },
    "rock": {
        "vpip": 0.10, "pfr": 0.08, "three_bet": 0.02, "fold_to_cbet": 0.65,
        "aggression_factor": 1.2, "wtsd": 0.18, "cbet_freq": 0.75,
        "check_raise_freq": 0.02, "overbet_freq": 0.00, "limp_freq": 0.01,
        "squeeze_freq": 0.01, "donk_freq": 0.01,
    },
    "calling_station": {
        "vpip": 0.45, "pfr": 0.08, "three_bet": 0.01, "fold_to_cbet": 0.20,
        "aggression_factor": 0.5, "wtsd": 0.45, "cbet_freq": 0.30,
        "check_raise_freq": 0.01, "overbet_freq": 0.00, "limp_freq": 0.20,
        "squeeze_freq": 0.01, "donk_freq": 0.08,
    },
}

# Skill-level adjustments (higher skill → tighter, more aggressive, more aware)
_SKILL_ADJUSTMENTS = {
    "vpip": -0.08,        # skilled players play fewer hands
    "pfr": +0.05,         # but raise more when they enter
    "three_bet": +0.03,
    "fold_to_cbet": -0.10,
    "aggression_factor": +0.5,
    "cbet_freq": +0.10,
    "check_raise_freq": +0.04,
    "limp_freq": -0.10,   # skilled players don't limp
    "donk_freq": -0.05,   # skilled players don't donk
}

_NAMES_FIRST = [
    "Alex", "Jordan", "Sam", "Taylor", "Morgan", "Casey", "Riley",
    "Quinn", "Avery", "Drew", "Hayden", "Reese", "Blake", "Sage",
    "Phoenix", "River", "Sky", "Storm", "Ember", "Frost",
]

_NAMES_LAST = [
    "Chen", "Kim", "Patel", "Garcia", "Müller", "Tanaka", "Silva",
    "Johnson", "Lee", "Kowalski", "Dubois", "Ivanov", "Santos",
    "Fischer", "Yamada", "Park", "Nguyen", "Hansen", "Costa", "Wolf",
]


class SyntheticPlayerGenerator:
    """Generates diverse synthetic player profiles.

    Uses a PGM-inspired approach: first samples high-level traits
    (skill, archetype, tilt propensity), then conditions detailed
    stats on those traits with controlled noise.

    Args:
        seed: Random seed for reproducibility.
        noise_scale: Amount of per-stat noise (0 = exact template).
    """

    def __init__(self, seed: int = 42, noise_scale: float = 0.15) -> None:
        self.rng = random.Random(seed)
        self.noise_scale = noise_scale

    def generate(self, archetype: str | None = None) -> SyntheticPlayer:
        """Generate a single synthetic player.

        Args:
            archetype: Force a specific archetype (random if None).

        Returns:
            SyntheticPlayer with full profile.
        """
        # Sample archetype
        if archetype is None:
            archetype = self.rng.choice(list(_ARCHETYPE_TEMPLATES.keys()))

        template = _ARCHETYPE_TEMPLATES.get(archetype, _ARCHETYPE_TEMPLATES["tag"])

        # Sample high-level traits
        skill_level = self.rng.betavariate(2.0, 3.0)  # skewed toward lower skill
        tilt_propensity = self.rng.betavariate(2.0, 5.0)  # most players aren't super tilty
        experience_hours = self.rng.expovariate(1.0 / 500.0)  # mean 500 hours

        # Generate stats from template + skill adjustments + noise
        stats = self._sample_stats(template, skill_level)

        # Timing from skill (skilled = faster, more consistent)
        stats.timing_mean_ms = max(500, 3000 - skill_level * 2000 + self.rng.gauss(0, 300))
        stats.timing_std_ms = max(100, 1500 - skill_level * 1000 + self.rng.gauss(0, 200))
        stats.positional_awareness = min(1.0, max(0.0, skill_level + self.rng.gauss(0, 0.1)))
        stats.sizing_precision = min(1.0, max(0.0, skill_level * 0.8 + self.rng.gauss(0, 0.1)))

        # Generate name
        first = self.rng.choice(_NAMES_FIRST)
        last = self.rng.choice(_NAMES_LAST)
        name = f"{first}_{last}_{self.rng.randint(100, 999)}"

        return SyntheticPlayer(
            name=name,
            archetype=archetype,
            skill_level=skill_level,
            tilt_propensity=tilt_propensity,
            experience_hours=round(experience_hours, 1),
            stats=stats,
        )

    def generate_batch(
        self,
        n: int,
        archetype_distribution: dict[str, float] | None = None,
    ) -> list[SyntheticPlayer]:
        """Generate a batch of diverse synthetic players.

        Args:
            n: Number of players to generate.
            archetype_distribution: Optional weights per archetype.

        Returns:
            List of SyntheticPlayer instances.
        """
        if archetype_distribution is None:
            # Default distribution (realistic poker ecosystem)
            archetype_distribution = {
                "fish": 0.25, "calling_station": 0.10, "whale": 0.05,
                "nit": 0.10, "rock": 0.05, "tag": 0.25,
                "lag": 0.12, "maniac": 0.08,
            }

        # Normalize
        total = sum(archetype_distribution.values())
        weights = {k: v / total for k, v in archetype_distribution.items()}

        players = []
        for _ in range(n):
            # Sample archetype from distribution
            r = self.rng.random()
            cumsum = 0.0
            archetype = "tag"
            for arch, w in weights.items():
                cumsum += w
                if r <= cumsum:
                    archetype = arch
                    break
            players.append(self.generate(archetype))

        return players

    def _sample_stats(
        self,
        template: dict[str, float],
        skill_level: float,
    ) -> SyntheticPlayerStats:
        """Sample stats from template with skill adjustment and noise."""
        stats = SyntheticPlayerStats()

        for stat_name, base_value in template.items():
            # Skill adjustment
            skill_adj = _SKILL_ADJUSTMENTS.get(stat_name, 0.0) * skill_level

            # Noise
            noise = self.rng.gauss(0, self.noise_scale * abs(base_value + 0.01))

            # Final value clamped to valid range
            value = base_value + skill_adj + noise

            if stat_name == "aggression_factor":
                value = max(0.1, min(10.0, value))
            else:
                value = max(0.0, min(1.0, value))

            setattr(stats, stat_name, round(value, 4))

        return stats
