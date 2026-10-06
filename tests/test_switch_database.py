from __future__ import annotations

from typing import Callable, Generator

import psycopg
import pytest
from conftest import TEST_DB_CONN
from harlequin.exception import HarlequinQueryError

from harlequin_postgres.adapter import MAX_OPEN_DATABASES, HarlequinPostgresConnection
from harlequin_postgres.catalog import DatabaseCatalogItem
from harlequin_postgres.interactions import execute_use_database


@pytest.fixture
def make_database() -> Generator[Callable[[str], None], None, None]:
    """Creates databases beside `test`, each with a table and a matview."""
    created: list[str] = []

    def _make(name: str) -> None:
        with psycopg.connect(conninfo=TEST_DB_CONN, dbname="postgres") as pgconn:
            pgconn.autocommit = True
            pgconn.execute(f'drop database if exists "{name}" with (force)')
            pgconn.execute(f'create database "{name}"')
        created.append(name)
        with psycopg.connect(conninfo=TEST_DB_CONN, dbname=name) as dbconn:
            dbconn.execute(f'create schema "{name}_schema"')
            dbconn.execute(f'create table "{name}_schema".t (id int)')
            dbconn.execute(
                f'create materialized view "{name}_schema".mv as select 1 as one'
            )

    yield _make

    with psycopg.connect(conninfo=TEST_DB_CONN, dbname="postgres") as pgconn:
        pgconn.autocommit = True
        for name in created:
            pgconn.execute(f'drop database if exists "{name}" with (force)')


def _database_item(
    connection: HarlequinPostgresConnection, name: str
) -> DatabaseCatalogItem:
    item = next(item for item in connection.get_catalog().items if item.label == name)
    assert isinstance(item, DatabaseCatalogItem)
    return item


class FakeDriver:
    def __init__(self) -> None:
        self.notifications: list[tuple[str, str]] = []
        self.refreshed = False

    def notify(self, message: str, severity: str = "information") -> None:
        self.notifications.append((message, severity))

    def refresh_catalog(self) -> None:
        self.refreshed = True


def test_other_database_shows_its_own_catalog(
    connection: HarlequinPostgresConnection,
    make_database: Callable[[str], None],
) -> None:
    make_database("other")
    other = _database_item(connection, "other")

    schemas = other.fetch_children()
    assert [schema.label for schema in schemas] == ["other_schema", "public"]

    relations = schemas[0].fetch_children()
    assert [(r.label, r.type_label) for r in relations] == [("t", "t"), ("mv", "mv")]
    assert [c.label for c in relations[0].fetch_children()] == ["id"]
    assert [c.label for c in relations[1].fetch_children()] == ["one"]

    # browsing does not move the editor's queries
    assert connection.active_database == "test"


def test_use_database_moves_queries_and_marks_it_active(
    connection: HarlequinPostgresConnection,
    make_database: Callable[[str], None],
) -> None:
    make_database("other")
    assert _database_item(connection, "test").type_label == "db ●"

    connection.use_database("other")

    cur = connection.execute("select current_database()")
    assert cur is not None
    assert cur.fetchall() == [("other",)]
    assert connection.active_database == "other"
    assert _database_item(connection, "other").type_label == "db ●"
    assert _database_item(connection, "test").type_label == "db"
    # the active database is read through the main pool, not a catalog pool
    assert "other" not in connection._catalog_pools
    # the catalog preloads the active database, which feeds autocomplete
    assert _database_item(connection, "other").loaded
    assert not _database_item(connection, "test").loaded


def test_use_database_keeps_the_transaction_mode(
    connection: HarlequinPostgresConnection,
    make_database: Callable[[str], None],
) -> None:
    make_database("other")
    connection.toggle_transaction_mode()
    assert connection.transaction_mode.label == "Manual"

    connection.use_database("other")
    connection.execute("create table manual_only (id int)")
    connection.rollback()

    cur = connection.execute("select to_regclass('manual_only')::text")
    assert cur is not None
    assert cur.fetchall() == [(None,)]


def test_use_database_refuses_while_a_transaction_is_open(
    connection: HarlequinPostgresConnection,
    make_database: Callable[[str], None],
) -> None:
    make_database("other")
    connection.toggle_transaction_mode()
    connection.execute("create table pending (id int)")

    with pytest.raises(HarlequinQueryError):
        connection.use_database("other")

    assert connection.active_database == "test"
    connection.rollback()


def test_catalog_keeps_at_most_max_open_databases(
    connection: HarlequinPostgresConnection,
    make_database: Callable[[str], None],
) -> None:
    names = [f"other_{i}" for i in range(MAX_OPEN_DATABASES + 1)]
    for name in names:
        make_database(name)

    for name in names:
        _database_item(connection, name).fetch_children()
        # the active database's main pool counts toward the limit
        assert len(connection._catalog_pools) <= MAX_OPEN_DATABASES - 1

    assert list(connection._catalog_pools) == names[-(MAX_OPEN_DATABASES - 1) :]
    # an evicted database reopens on demand
    schemas = _database_item(connection, names[0]).fetch_children()
    assert schemas[0].label == f"{names[0]}_schema"


def test_use_database_interaction(
    connection: HarlequinPostgresConnection,
    make_database: Callable[[str], None],
) -> None:
    make_database("other")
    driver = FakeDriver()

    execute_use_database(_database_item(connection, "other"), driver)  # type: ignore[arg-type]

    assert connection.active_database == "other"
    assert driver.notifications == [("Queries now run on other", "information")]
    assert driver.refreshed


def test_use_database_interaction_reports_failure(
    connection: HarlequinPostgresConnection,
    make_database: Callable[[str], None],
) -> None:
    make_database("other")
    connection.toggle_transaction_mode()
    connection.execute("select 1")
    driver = FakeDriver()

    execute_use_database(_database_item(connection, "other"), driver)  # type: ignore[arg-type]

    assert connection.active_database == "test"
    assert driver.notifications[0][1] == "error"
    assert not driver.refreshed
    connection.rollback()
