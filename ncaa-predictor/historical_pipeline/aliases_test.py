from historical_pipeline.aliases import AliasResolver


def test_alias_resolver_handles_common_tournament_name_collisions() -> None:
    resolver = AliasResolver()

    saint_marys = resolver.resolve("Saint Mary's")
    uconn = resolver.resolve("Connecticut")
    nc_state = resolver.resolve("N.C. State")
    miami_fl = resolver.resolve("Miami (FL)")
    miami_ambiguous = resolver.resolve("Miami")
    ole_miss = resolver.resolve("Mississippi")

    assert saint_marys.canonical_team_id == "saint-marys"
    assert uconn.canonical_team_id == "connecticut"
    assert nc_state.canonical_team_id == "nc-state"
    assert miami_fl.canonical_team_id == "miami-fl"
    assert miami_ambiguous.needs_review is True
    assert ole_miss.canonical_team_id == "mississippi"
