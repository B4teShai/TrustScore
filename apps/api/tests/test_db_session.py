from app.db import session


def test_invalid_database_url_does_not_crash_scans() -> None:
    original_url = session.settings.database_url
    session.get_engine.cache_clear()
    try:
        object.__setattr__(session.settings, "database_url", "not a sqlalchemy url")

        assert session.get_engine() is None
    finally:
        object.__setattr__(session.settings, "database_url", original_url)
        session.get_engine.cache_clear()
