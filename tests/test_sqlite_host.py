from hosts.sqlite_host import SQLiteExecutor
from natlang.execution import CrispRequest, ExecutionError


def request(sql, args=None):
    return CrispRequest(sql, {"args": args or {}}, False, "eval")


def test_join_two_tables_with_bound_parameters():
    with SQLiteExecutor() as db:
        db.prepare_schema("CREATE TABLE people(id INTEGER, name TEXT);"
                          "CREATE TABLE sales(person_id INTEGER, amount REAL);"
                          "INSERT INTO people VALUES (1, 'Dana'), (2, 'Lee');"
                          "INSERT INTO sales VALUES (1, 12.5), (2, 8.0);")
        rows = db.run(request("SELECT name, amount FROM people JOIN sales ON people.id=sales.person_id "
                              "WHERE amount >= :floor ORDER BY name", {"floor": 10}), None)
        assert rows == [{"name": "Dana", "amount": 12.5}]


def test_patch_transaction_preview_and_rollback():
    with SQLiteExecutor() as db:
        db.prepare_schema("CREATE TABLE items(id INTEGER PRIMARY KEY, quantity INTEGER);"
                          "INSERT INTO items VALUES (1, 3);")
        db.begin()
        changed = db.run(request("UPDATE items SET quantity=:quantity WHERE id=:id",
                                 {"quantity": 5, "id": 1}), None)
        assert changed["rows_affected"] == 1
        assert db.run(request("SELECT quantity FROM items WHERE id=1"), None) == [{"quantity": 5}]
        db.rollback()
        assert db.run(request("SELECT quantity FROM items WHERE id=1"), None) == [{"quantity": 3}]
        assert [e["operation"] for e in db.drain_events()].count("sql.rollback") == 1


def test_native_blob_is_rejected_at_value_boundary():
    with SQLiteExecutor() as db:
        db.prepare_schema("CREATE TABLE bytes(value BLOB); INSERT INTO bytes VALUES (X'01');")
        try:
            db.run(request("SELECT value FROM bytes"), None)
        except ExecutionError:
            pass
        else:
            raise AssertionError("binary value crossed the typed boundary")
