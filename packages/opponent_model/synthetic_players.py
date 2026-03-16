"""Synthetic Player Generator — Probabilistic model for diverse opponent profiles.

Generates realistic synthetic opponent profiles using a probabilistic
graphical model (PGM) approach inspired by NVIDIA's Nemotron-Personas-Singapore.

Architecture follows a hierarchical conditional dependency graph:

Layer 1 (Sampled first):
    personality (Big Five) → archetype probabilities
    risk_tolerance → aggression, overbet_freq

Layer 2 (Conditioned on Layer 1):
    archetype → base stats template
    skill_level | personality.conscientiousness → stat adjustments

Layer 3 (Conditioned on Layers 1-2):
    stats = template + skill_adj + personality_adj + noise
    timing = f(skill, personality.neuroticism)
    tilt_propensity = f(neuroticism, agreeableness)

Layer 4 (Session dynamics):
    session_state → stat drift (tilt, fatigue, momentum)

References:
- NVIDIA Nemotron-Personas (2026): PGM grounded in census distributions
- DeepPersona (2511.07338): 100+ hierarchical attributes per persona
- SCOPE (2601.07110): demographics = 1.5% variance; psychology dominates
- PSYDIAL (2024): Big Five personality → behavioral generation

Usage::

    generator = SyntheticPlayerGenerator(seed=42)
    profiles = generator.generate_batch(100)
    for p in profiles:
        print(p.name, p.archetype, p.personality, p.stats)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class PersonalityProfile:
    """Big Five personality traits mapped to poker behavior.

    Each trait is 0.0-1.0. Trait-to-poker mappings based on PSYDIAL
    and behavioral psychology research:

    - openness: creative play, unusual lines, bluff frequency
    - conscientiousness: disciplined play, bankroll management, GTO adherence
    - extraversion: aggression, table talk, action-seeking
    - agreeableness: passive play, calling frequency, avoid confrontation
    - neuroticism: tilt propensity, timing variance, emotional swings
    """

    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5

    def to_vector(self) -> list[float]:
        """5-dim personality vector."""
        return [self.openness, self.conscientiousness, self.extraversion,
                self.agreeableness, self.neuroticism]

    @property
    def risk_tolerance(self) -> float:
        """Derived: openness + extraversion - conscientiousness - agreeableness."""
        raw = (self.openness + self.extraversion -
               self.conscientiousness - self.agreeableness + 1.0) / 3.0
        return max(0.0, min(1.0, raw))

    @property
    def emotional_stability(self) -> float:
        """Inverse of neuroticism."""
        return 1.0 - self.neuroticism


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
    personality: PersonalityProfile = field(default_factory=PersonalityProfile)


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


# Personality → Archetype affinity weights (Big Five → archetype probability boost)
# Based on PSYDIAL (2024) and behavioral psychology research
_PERSONALITY_ARCHETYPE_AFFINITY: dict[str, dict[str, float]] = {
    "nit": {"openness": -0.3, "conscientiousness": 0.5, "extraversion": -0.3, "agreeableness": 0.1, "neuroticism": 0.2},
    "tag": {"openness": 0.1, "conscientiousness": 0.4, "extraversion": 0.1, "agreeableness": -0.1, "neuroticism": -0.1},
    "lag": {"openness": 0.4, "conscientiousness": 0.2, "extraversion": 0.4, "agreeableness": -0.3, "neuroticism": 0.0},
    "maniac": {"openness": 0.3, "conscientiousness": -0.4, "extraversion": 0.5, "agreeableness": -0.5, "neuroticism": 0.3},
    "fish": {"openness": 0.0, "conscientiousness": -0.3, "extraversion": 0.2, "agreeableness": 0.3, "neuroticism": 0.1},
    "whale": {"openness": 0.2, "conscientiousness": -0.2, "extraversion": 0.3, "agreeableness": 0.1, "neuroticism": 0.0},
    "rock": {"openness": -0.4, "conscientiousness": 0.5, "extraversion": -0.4, "agreeableness": 0.2, "neuroticism": 0.3},
    "calling_station": {"openness": -0.1, "conscientiousness": -0.2, "extraversion": 0.0, "agreeableness": 0.5, "neuroticism": 0.1},
}

# Personality adjustments to stats (on top of skill adjustments)
_PERSONALITY_STAT_EFFECTS: dict[str, dict[str, float]] = {
    "openness": {"overbet_freq": 0.04, "check_raise_freq": 0.03, "squeeze_freq": 0.02},
    "conscientiousness": {"vpip": -0.05, "limp_freq": -0.05, "donk_freq": -0.03, "cbet_freq": 0.05},
    "extraversion": {"vpip": 0.05, "pfr": 0.03, "three_bet": 0.02, "aggression_factor": 0.5},
    "agreeableness": {"fold_to_cbet": 0.08, "aggression_factor": -0.5, "wtsd": 0.05},
    "neuroticism": {"overbet_freq": 0.03, "timing_mean_ms": 500.0, "timing_std_ms": 300.0},
}


class SyntheticPlayerGenerator:
    """Generates diverse synthetic player profiles using hierarchical PGM.

    Generation follows a 3-layer conditional dependency graph
    inspired by NVIDIA Nemotron-Personas (2026):

    Layer 1: Sample personality (Big Five traits)
    Layer 2: personality → archetype probabilities → skill_level
    Layer 3: archetype + skill + personality → stats + timing

    Args:
        seed: Random seed for reproducibility.
        noise_scale: Amount of per-stat noise (0 = exact template).
    """

    def __init__(self, seed: int = 42, noise_scale: float = 0.15) -> None:
        self.rng = random.Random(seed)
        self.noise_scale = noise_scale

    def _sample_personality(self) -> PersonalityProfile:
        """Sample Big Five personality traits from population distributions.

        Uses beta distributions calibrated to real-world trait distributions
        (SCOPE Framework, 2601.07110).
        """
        return PersonalityProfile(
            openness=self.rng.betavariate(4.0, 4.0),        # centered ~0.5
            conscientiousness=self.rng.betavariate(5.0, 3.0),  # skewed high
            extraversion=self.rng.betavariate(3.0, 3.0),     # centered
            agreeableness=self.rng.betavariate(5.0, 3.0),    # skewed high
            neuroticism=self.rng.betavariate(2.5, 4.0),      # skewed low
        )

    def _archetype_from_personality(
        self, personality: PersonalityProfile
    ) -> str:
        """Conditionally sample archetype given personality (PGM Layer 2).

        Computes affinity scores between personality and each archetype,
        then samples from softmax distribution.
        """
        scores: dict[str, float] = {}
        for arch, affinities in _PERSONALITY_ARCHETYPE_AFFINITY.items():
            score = 1.0  # base weight
            for trait, weight in affinities.items():
                trait_val = getattr(personality, trait)
                score += weight * (trait_val - 0.5) * 2.0  # center and scale
            scores[arch] = max(0.01, score)

        # Softmax sampling
        total = sum(scores.values())
        r = self.rng.random() * total
        cumsum = 0.0
        for arch, s in scores.items():
            cumsum += s
            if r <= cumsum:
                return arch
        return "tag"

    def generate(self, archetype: str | None = None) -> SyntheticPlayer:
        """Generate a single synthetic player via hierarchical PGM.

        PGM Layer 1: Sample personality
        PGM Layer 2: personality → archetype + skill
        PGM Layer 3: archetype + skill + personality → stats

        Args:
            archetype: Force a specific archetype (random if None).

        Returns:
            SyntheticPlayer with full profile including personality.
        """
        # Layer 1: Sample personality
        personality = self._sample_personality()

        # Layer 2: Condition archetype on personality
        if archetype is None:
            archetype = self._archetype_from_personality(personality)

        template = _ARCHETYPE_TEMPLATES.get(archetype, _ARCHETYPE_TEMPLATES["tag"])

        # Skill conditioned on conscientiousness (disciplined → higher skill)
        skill_base = self.rng.betavariate(2.0, 3.0)
        skill_level = min(1.0, max(0.0,
            skill_base * 0.7 + personality.conscientiousness * 0.3
        ))

        # Tilt conditioned on neuroticism (SCOPE insight: psychology > demographics)
        tilt_propensity = min(1.0, max(0.0,
            personality.neuroticism * 0.6 +
            (1.0 - personality.agreeableness) * 0.2 +
            self.rng.gauss(0, 0.1)
        ))

        experience_hours = self.rng.expovariate(1.0 / 500.0)

        # Layer 3: Generate stats conditioned on all higher layers
        stats = self._sample_stats(template, skill_level)
        self._apply_personality_effects(stats, personality)

        # Timing conditioned on skill + neuroticism
        stats.timing_mean_ms = max(500,
            3000 - skill_level * 2000 +
            personality.neuroticism * 500 +
            self.rng.gauss(0, 300)
        )
        stats.timing_std_ms = max(100,
            1500 - skill_level * 1000 +
            personality.neuroticism * 400 +
            self.rng.gauss(0, 200)
        )
        stats.positional_awareness = min(1.0, max(0.0,
            skill_level * 0.7 + personality.conscientiousness * 0.3 +
            self.rng.gauss(0, 0.1)
        ))
        stats.sizing_precision = min(1.0, max(0.0,
            skill_level * 0.6 + personality.conscientiousness * 0.2 +
            self.rng.gauss(0, 0.1)
        ))

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
            personality=personality,
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

    def _apply_personality_effects(
        self,
        stats: SyntheticPlayerStats,
        personality: PersonalityProfile,
    ) -> None:
        """Apply personality-conditioned adjustments to stats (PGM Layer 3).

        Each Big Five trait has specific effects on poker stats,
        scaled by trait deviation from population mean (0.5).
        """
        for trait_name, effects in _PERSONALITY_STAT_EFFECTS.items():
            trait_val = getattr(personality, trait_name)
            deviation = trait_val - 0.5  # how far from average

            for stat_name, effect_magnitude in effects.items():
                current = getattr(stats, stat_name)
                adjustment = effect_magnitude * deviation

                new_val = current + adjustment

                if stat_name == "aggression_factor":
                    new_val = max(0.1, min(10.0, new_val))
                elif stat_name in ("timing_mean_ms", "timing_std_ms"):
                    new_val = max(100.0, new_val)
                else:
                    new_val = max(0.0, min(1.0, new_val))

                setattr(stats, stat_name, round(new_val, 4))
