"""Fast idiom detection via regex pattern matching (CPU, <10ms)."""

import re
import json
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class IdiomMatch:
    canonical: str
    literal: str
    meaning: str
    region: str = "universal"
    frequency: str = "common"
    match_start: int = 0
    match_end: int = 0


@dataclass
class IdiomPattern:
    pattern: re.Pattern
    canonical: str
    literal: str
    meaning: str
    region: str = "universal"
    frequency: str = "common"


class IdiomScanner:
    """Scans transcripts for known idioms using compiled regex patterns.

    Patterns are loaded from two sources:
    - JSON files in data/idioms/ (loaded at startup via load_from_json)
    - Database idiom_patterns table (loaded via load_from_db)

    Both load methods silently skip entries with invalid regex (re.error).
    scan() is CPU-only and runs in <10ms. Deduplicates by canonical form.
    """

    def __init__(self):
        self.patterns: list[IdiomPattern] = []

    def load_from_json(self, path: Path):
        """Load idiom patterns from a JSON file."""
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            try:
                compiled = re.compile(entry["pattern"], re.IGNORECASE)
                self.patterns.append(IdiomPattern(
                    pattern=compiled,
                    canonical=entry["canonical"],
                    literal=entry.get("literal", ""),
                    meaning=entry["meaning"],
                    region=entry.get("region", "universal"),
                    frequency=entry.get("frequency", "common"),
                ))
            except re.error:
                continue  # skip bad patterns

    def load_from_db(self, rows: list[dict]):
        """Load patterns from database rows."""
        for row in rows:
            try:
                compiled = re.compile(row["pattern"], re.IGNORECASE)
                self.patterns.append(IdiomPattern(
                    pattern=compiled,
                    canonical=row["canonical"],
                    literal=row.get("literal", ""),
                    meaning=row["meaning"],
                    region=row.get("region", "universal"),
                    frequency=row.get("frequency", "common"),
                ))
            except re.error:
                continue

    def scan(self, text: str) -> list[IdiomMatch]:
        """Scan text for matching idiom patterns. Fast — runs in <10ms."""
        matches = []
        seen_canonicals = set()

        for ip in self.patterns:
            m = ip.pattern.search(text)
            if m and ip.canonical.lower() not in seen_canonicals:
                seen_canonicals.add(ip.canonical.lower())
                matches.append(IdiomMatch(
                    canonical=ip.canonical,
                    literal=ip.literal,
                    meaning=ip.meaning,
                    region=ip.region,
                    frequency=ip.frequency,
                    match_start=m.start(),
                    match_end=m.end(),
                ))

        return matches

    @property
    def count(self) -> int:
        return len(self.patterns)


def create_starter_idioms() -> list[dict]:
    """Return a starter set of common Spanish idioms with regex patterns."""
    return [
        {
            "pattern": r"importa[rn]?\s+(un\s+)?pepino",
            "canonical": "importar un pepino",
            "literal": "to matter a cucumber",
            "meaning": "to not care at all",
            "region": "universal",
            "frequency": "very common"
        },
        {
            "pattern": r"est[aá][rs]?\s+en\s+las\s+nubes",
            "canonical": "estar en las nubes",
            "literal": "to be in the clouds",
            "meaning": "to be daydreaming, absent-minded",
            "region": "universal",
            "frequency": "very common"
        },
        {
            "pattern": r"tomar\s+(el\s+)?pelo",
            "canonical": "tomar el pelo",
            "literal": "to take the hair",
            "meaning": "to pull someone's leg, to joke with someone",
            "region": "universal",
            "frequency": "very common"
        },
        {
            "pattern": r"meter\s+la\s+pata",
            "canonical": "meter la pata",
            "literal": "to put the paw in",
            "meaning": "to put your foot in it, to make a blunder",
            "region": "universal",
            "frequency": "very common"
        },
        {
            "pattern": r"no\s+tiene[sn]?\s+pelos?\s+en\s+la\s+lengua",
            "canonical": "no tener pelos en la lengua",
            "literal": "to not have hairs on the tongue",
            "meaning": "to not mince words, to speak frankly",
            "region": "universal",
            "frequency": "common"
        },
        {
            "pattern": r"dar\s+(en\s+)?el\s+clavo",
            "canonical": "dar en el clavo",
            "literal": "to hit the nail",
            "meaning": "to hit the nail on the head, to get it right",
            "region": "universal",
            "frequency": "common"
        },
        {
            "pattern": r"echar\s+(una\s+)?mano",
            "canonical": "echar una mano",
            "literal": "to throw a hand",
            "meaning": "to lend a hand, to help out",
            "region": "universal",
            "frequency": "very common"
        },
        {
            "pattern": r"tirar\s+la\s+toalla",
            "canonical": "tirar la toalla",
            "literal": "to throw the towel",
            "meaning": "to throw in the towel, to give up",
            "region": "universal",
            "frequency": "common"
        },
        {
            "pattern": r"costar\s+un\s+ojo\s+de\s+la\s+cara",
            "canonical": "costar un ojo de la cara",
            "literal": "to cost an eye from the face",
            "meaning": "to cost an arm and a leg, very expensive",
            "region": "universal",
            "frequency": "common"
        },
        {
            "pattern": r"tener?\s+(mucha\s+)?cara(\s+dura)?",
            "canonical": "tener cara dura",
            "literal": "to have a hard face",
            "meaning": "to have a lot of nerve, to be shameless",
            "region": "spain",
            "frequency": "very common"
        },
        {
            "pattern": r"quedarse?\s+(de\s+)?piedra",
            "canonical": "quedarse de piedra",
            "literal": "to remain as stone",
            "meaning": "to be stunned, shocked",
            "region": "spain",
            "frequency": "common"
        },
        {
            "pattern": r"flipar(\s+en\s+colores)?",
            "canonical": "flipar en colores",
            "literal": "to flip out in colors",
            "meaning": "to be amazed, mind-blown (informal Spain)",
            "region": "spain",
            "frequency": "common"
        },
        {
            "pattern": r"mola[rs]?\s+(mucho|mazo|mogollón)?",
            "canonical": "molar",
            "literal": "to molar (slang verb)",
            "meaning": "to be cool, awesome (Spain slang)",
            "region": "spain",
            "frequency": "very common"
        },
        {
            "pattern": r"(está[rs]?\s+)?hasta\s+las\s+narices",
            "canonical": "estar hasta las narices",
            "literal": "to be up to the nostrils",
            "meaning": "to be fed up, sick and tired of something",
            "region": "spain",
            "frequency": "very common"
        },
        {
            "pattern": r"ir\s+(de\s+)?culo",
            "canonical": "ir de culo",
            "literal": "to go on one's backside",
            "meaning": "to be in a rush / in trouble (Spain, vulgar-ish)",
            "region": "spain",
            "frequency": "common"
        },
        {
            "pattern": r"(dar|dando)\s+(la\s+)?lata",
            "canonical": "dar la lata",
            "literal": "to give the can",
            "meaning": "to be annoying, to pester",
            "region": "spain",
            "frequency": "common"
        },
        {
            "pattern": r"(es\s+)?pan\s+comido",
            "canonical": "ser pan comido",
            "literal": "to be eaten bread",
            "meaning": "to be a piece of cake, very easy",
            "region": "universal",
            "frequency": "common"
        },
        {
            "pattern": r"llover\s+(a\s+)?cántaros",
            "canonical": "llover a cántaros",
            "literal": "to rain pitchers",
            "meaning": "to rain cats and dogs, to pour",
            "region": "universal",
            "frequency": "common"
        },
        {
            "pattern": r"(no\s+)?pegar\s+ojo",
            "canonical": "no pegar ojo",
            "literal": "to not stick an eye",
            "meaning": "to not sleep a wink",
            "region": "universal",
            "frequency": "common"
        },
        {
            "pattern": r"ponerse?\s+las\s+pilas",
            "canonical": "ponerse las pilas",
            "literal": "to put on the batteries",
            "meaning": "to get one's act together, to buckle down",
            "region": "universal",
            "frequency": "very common"
        },
        {
            "pattern": r"(no\s+)?tener?\s+ni\s+(idea|puta\s+idea|pajolera\s+idea)",
            "canonical": "no tener ni idea",
            "literal": "to not have even an idea",
            "meaning": "to have absolutely no clue",
            "region": "universal",
            "frequency": "very common"
        },
        {
            "pattern": r"(ser|es)\s+un\s+borde",
            "canonical": "ser un borde",
            "literal": "to be an edge",
            "meaning": "to be rude, unfriendly (Spain)",
            "region": "spain",
            "frequency": "common"
        },
        {
            "pattern": r"mala\s+leche",
            "canonical": "mala leche",
            "literal": "bad milk",
            "meaning": "bad temper, bad intentions",
            "region": "spain",
            "frequency": "very common"
        },
        {
            "pattern": r"(ser|es)\s+la\s+caña",
            "canonical": "ser la caña",
            "literal": "to be the cane/reed",
            "meaning": "to be amazing/awesome (Spain)",
            "region": "spain",
            "frequency": "common"
        },
        {
            "pattern": r"(tener?|tengo|tienes?)\s+(mucha\s+)?morriña",
            "canonical": "tener morriña",
            "literal": "to have morriña",
            "meaning": "to be homesick (Galician origin, used across Spain)",
            "region": "spain",
            "frequency": "common"
        },
    ]
