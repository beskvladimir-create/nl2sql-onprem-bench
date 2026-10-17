from nl2sql_bench.textfix import repair_ws


def test_repairs_glued_keywords_outside_quotes():
    assert repair_ws("WHERE x >500AND y = 'a2AND' LIMIT1") == "WHERE x >500 AND y = 'a2AND' LIMIT 1"


def test_leaves_valid_sql_and_identifiers_alone():
    for s in ["SELECT T1.id FROM t AS T1 WHERE T1.x = 2 ORDER BY 1", 'SELECT "col2AND" FROM t2']:
        assert repair_ws(s) == s


def test_code_fence_is_not_a_quote():
    raw = "```sql\nSELECT a FROM t WHERE x >500AND y = 1 LIMIT1\n```"
    assert repair_ws(raw) == "```sql\nSELECT a FROM t WHERE x >500 AND y = 1 LIMIT 1\n```"
