"""סדר וכיוון הזוגות בשאלון הסוכנים: תשובה על זוג הפוך חייבת לחזור לסדר הקנוני עם היפוך V/A."""
import contextlib, io, os, sys

import pytest


@pytest.fixture(scope='module')
def app(tmp_path_factory):
    os.environ['ISM_DB_PATH'] = str(tmp_path_factory.mktemp('db') / 'p.db')
    with contextlib.redirect_stderr(io.StringIO()):
        import project2025
    return project2025


F = ['a', 'b', 'c']


def test_canonical_keeps_forward_pairs(app):
    assert app.canonical_relation('a', 'c', 'V', F) == ('a', 'c', 'V')


def test_canonical_flips_reversed_pairs(app):
    assert app.canonical_relation('c', 'a', 'V', F) == ('a', 'c', 'A')
    assert app.canonical_relation('c', 'a', 'A', F) == ('a', 'c', 'V')
    assert app.canonical_relation('b', 'a', 'X', F) == ('a', 'b', 'X')


def test_canonical_rejects_unknown_symbols(app):
    assert app.canonical_relation('a', 'b', 'Q', F) == ('a', 'b', 'O')


def test_shuffled_prompt_answers_map_back(app, monkeypatch):
    """הסוכן מקבל זוגות בסדר ובכיוון אקראיים; כל תשובה חוזרת לתא הנכון במטריצה."""
    import json, re
    truth = {('a', 'b'): 'V', ('a', 'c'): 'O', ('b', 'c'): 'A'}

    def fake_json(prompt, system, **k):
        out = []
        for x, y in re.findall(r'"([abc])"  מול  "([abc])"', prompt):
            if (x, y) in truth:
                rel = truth[(x, y)]
            else:
                rel = {'V': 'A', 'A': 'V', 'X': 'X', 'O': 'O'}[truth[(y, x)]]
            out.append({'factor_i': x, 'factor_j': y, 'relation': rel, 'justification': ''})
        return out, []

    monkeypatch.setattr(app, 'call_ai_json', fake_json)
    for seed in range(5):
        matrix, _ = app._simulate_synthetic_expert({'role': 'r'}, F, shuffle=True, seed=seed)
        got = {(i, j): matrix[i][j]['relation'] for i in matrix for j in matrix[i]}
        assert got == truth
