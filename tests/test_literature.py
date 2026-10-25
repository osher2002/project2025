"""
אימות מול ספרות: ה-SSIM של המומחים ב-Ahmad et al. (2019), Applied Sciences 9(2), 233 (טבלה 3), והתוצאות שפורסמו
(טבלה A2, איור 4, סעיף 4.2). המנוע חייב לשחזר אותן במדויק.
"""
import pandas as pd

from ism_core import transitive_closure, compute_ism_levels, classify_micmac

CODES = [f'F{k}' for k in range(1, 17)]
SSIM_ROWS = {
    1: 'O V V O O O O O O O O O O O V', 2: 'V V O O O O O O O O O O O V', 3: 'O O O O O O O O O O O O V',
    4: 'O A O O O O O O O O O V', 5: 'A V V V V V O O O O V', 6: 'V V V V V V O O O V', 7: 'V V V V O O O O V',
    8: 'V X V A A A A V', 9: 'A O O A A A V', 10: 'V A A A A V', 11: 'A A A O V', 12: 'A A O V', 13: 'O O V',
    14: 'O V', 15: 'V',
}
DRIVING = [4, 4, 2, 2, 7, 10, 6, 5, 2, 5, 2, 6, 7, 7, 6, 1]
DEPENDENCE = [1, 1, 3, 4, 2, 1, 3, 9, 10, 9, 10, 4, 1, 1, 1, 16]
LEVELS = [['F16'], ['F3', 'F4', 'F9', 'F11'], ['F1', 'F2', 'F8', 'F10'], ['F7', 'F12', 'F15'], ['F5', 'F13', 'F14'], ['F6']]
CLUSTERS = {'Driving': {'F6'}, 'Dependent': {'F8', 'F9', 'F10', 'F11', 'F16'}, 'Linkage': set()}


def irm_from_ssim():
    irm = pd.DataFrame(0, index=CODES, columns=CODES)
    for f in CODES:
        irm.loc[f, f] = 1
    for i, row in SSIM_ROWS.items():
        for k, s in enumerate(row.split()):
            a, b = CODES[i - 1], CODES[i + k]
            if s in 'VX':
                irm.loc[a, b] = 1
            if s in 'AX':
                irm.loc[b, a] = 1
    return irm


def test_reproduces_published_reachability_and_powers():
    frm = transitive_closure(irm_from_ssim())
    assert list(frm.sum(axis=1)) == DRIVING
    assert list(frm.sum(axis=0)) == DEPENDENCE


def test_reproduces_published_levels():
    levels = compute_ism_levels(transitive_closure(irm_from_ssim()))
    assert [sorted(l) for l in levels] == [sorted(l) for l in LEVELS]


def test_midpoint_threshold_reproduces_published_clusters():
    frm = transitive_closure(irm_from_ssim())
    cls = classify_micmac(frm.sum(axis=1), frm.sum(axis=0), threshold=len(CODES) / 2)['Classification']
    for name, members in CLUSTERS.items():
        assert {f for f in CODES if cls[f].startswith(name)} == members
