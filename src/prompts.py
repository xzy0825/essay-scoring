"""Map an essay to a PERSUADE-style prompt with distinctive phrases.

Student spelling is uneven, so a few common misspellings are included.
Essays that match none of the phrases stay "unknown" and use the global thresholds.
"""

from __future__ import annotations

import numpy as np

RULES: list[tuple[str, tuple[str, ...]]] = [
    ("electoral", ("electoral", "popular vote")),
    ("cowboys", ("cowboy", "seagoing", "unrra", "luke")),
    ("venus", ("venus",)),
    ("mars", ("mars", "landform")),
    ("facial", ("facial", "facs", "emotion")),
    ("driverless", ("driverless", "driveless", "self-driving", "self driving", "drive themselves")),
    ("carfree", ("car-free", "car free", "limiting car", "limit car", "cars emit")),
    ("community", ("community service",)),
    ("distance", ("distance learning", "online class", "stay home")),
    ("phones", ("cell phone", "cellphone", "mobile phone")),
    ("extra", ("extracurricular",)),
    ("summer", ("summer project", "summer break", "this summer")),
    ("opinions", ("multiple opinion", "second opinion", "another opinion")),
]


def prompt_of(text: str) -> str:
    lowered = text.lower()
    for name, keys in RULES:
        if any(key in lowered for key in keys):
            return name
    return "unknown"


def assign_prompts(texts) -> np.ndarray:
    return np.asarray([prompt_of(text) for text in texts], dtype=object)
