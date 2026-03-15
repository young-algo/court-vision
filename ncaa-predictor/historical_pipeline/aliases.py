from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .archive_data import archive_aliases_frame, has_archive_bundle
from .common import ALIASES_PATH, slugify_team
from .kaggle_data import has_kaggle_bundle, kaggle_aliases_frame


@dataclass
class AliasMatch:
    canonical_team_id: str | None
    canonical_team_name: str | None
    match_method: str
    needs_review: bool


class AliasResolver:
    def __init__(self) -> None:
        self.aliases = pd.read_csv(ALIASES_PATH)
        if has_archive_bundle():
            archive_aliases = archive_aliases_frame()
            self.aliases = pd.concat([self.aliases, archive_aliases], ignore_index=True)
        if has_kaggle_bundle():
            kaggle_aliases = kaggle_aliases_frame()
            self.aliases = pd.concat([self.aliases, kaggle_aliases], ignore_index=True)
            self.aliases = self.aliases.drop_duplicates(
                subset=["source_team_name"],
                keep="first",
            )
        self.lookup = {
            row.source_team_name.lower(): AliasMatch(
                canonical_team_id=row.canonical_team_id,
                canonical_team_name=row.canonical_team_name,
                match_method=row.match_method,
                needs_review=bool(row.needs_review),
            )
            for row in self.aliases.itertuples()
        }

    def resolve(self, team_name: str) -> AliasMatch:
        direct = self.lookup.get(team_name.lower())
        if direct:
            return direct

        return AliasMatch(
            canonical_team_id=slugify_team(team_name),
            canonical_team_name=team_name,
            match_method="generated_slug",
            needs_review=True,
        )

    def resolve_frame(self, frame: pd.DataFrame, source_column: str) -> pd.DataFrame:
        matches = frame[source_column].map(self.resolve)
        return frame.assign(
            canonical_team_id=matches.map(lambda match: match.canonical_team_id),
            canonical_team_name=matches.map(lambda match: match.canonical_team_name),
            match_method=matches.map(lambda match: match.match_method),
            needs_review=matches.map(lambda match: match.needs_review),
        )
