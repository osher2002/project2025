"""
סימולציה מלאה מקצה לקצה של מערכת ISM-MICMAC עם מנוע ה-AI האמיתי, ובדיקת ביצועים.

מריץ את הפונקציות של האפליקציה עצמה (לא העתק שלהן), באותו סדר שמנהל עובד בממשק:
  1. חקר הנושא והמלצה על גורמים (AI + חיפוש)
  2. המלצה על סוגי מומחים (AI)
  3. הפעלת סוכני מומחה סינתטיים שממלאים את שאלון ה-SSIM (AI + חיפוש), ושמירה במסד
  4. אגרגציה וזיהוי קונפליקטים
  5. הכרעת קונפליקטים ע"י שופט ה-AI, וחישוב IRM/FRM/רמות/MICMAC (קלאסי, Godet, Katz)
  6. דוח Excel מלא של האפליקציה
ובנוסף בדיקת ביצועים של מנוע החישוב (ללא AI) לגדלים שונים.

המפתח נקרא מ-.streamlit/secrets.toml (ANTHROPIC_API_KEY) ואינו מודפס. הנתונים נכתבים למסד נפרד של הריצה,
לא למסד המקומי של האפליקציה. הפלט: sim_runs/<זמן>/ עם report.md, report.json ו-ISM_Report.xlsx.

  python scripts/simulate.py                 # ריצה מלאה (עולה כסף - ראה הערכת עלות בתחילת הריצה)
  python scripts/simulate.py --dry-run       # בדיקת התשתית עם AI מדומה, ללא עלות וללא רשת
"""
import argparse
import contextlib
import datetime
import io
import itertools
import json
import os
import random
import sys
import time
import tomllib
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--topic', default='הפחתת תאונות דרכים בישראל')
    p.add_argument('--factors', type=int, default=8, help='כמות גורמים שה-AI ימליץ עליהם')
    p.add_argument('--expert-types', type=int, default=4, help='כמות סוגי מומחים')
    p.add_argument('--per-type', type=int, default=2, help='כמות סוכנים מכל סוג')
    p.add_argument('--out', default=os.path.join(ROOT, 'sim_runs'))
    p.add_argument('--dry-run', action='store_true', help='AI מדומה: בודק את התשתית בלי עלות ובלי רשת')
    return p.parse_args()


def load_secrets():
    path = os.path.join(ROOT, '.streamlit', 'secrets.toml')
    if not os.path.exists(path):
        return {}
    with open(path, 'rb') as f:
        return tomllib.load(f)


# ----------------------------------------------------------------------------- AI מדומה ל---dry-run
def install_fake_ai(app):
    rnd = random.Random(7)

    def fake_call(full_prompt, system_context):
        time.sleep(0.01)
        if 'RECOMMENDATION' in full_prompt:
            text = f"RECOMMENDATION: {rnd.choice('VAXO')}\nRATIONALE: הכרעה מדומה"
        elif 'תפקידי מומחים' in full_prompt:
            text = json.dumps([{"role": f"מומחה {k}", "expertise": "תחום", "rationale": "נימוק"} for k in range(1, 7)], ensure_ascii=False)
        elif 'גורמים מרכזיים' in full_prompt:
            text = json.dumps([{"name": f"גורם {k}", "description": "תיאור", "rationale": "נימוק"} for k in range(1, 9)], ensure_ascii=False)
        elif 'זוגות גורמים' in full_prompt:
            pairs = [line.split('"')[1::2] for line in full_prompt.splitlines() if 'מול' in line and line[:1].isdigit()]
            text = json.dumps([{"factor_i": a, "factor_j": b, "relation": rnd.choice('VAXOOOO'), "justification": "נימוק מדומה"}
                               for a, b in pairs], ensure_ascii=False)
        else:
            text = 'תשובה מדומה'
        app.record_ai_usage(len(full_prompt) // 3, len(text) // 3, model='dry-run', seconds=0.01, stop_reason='end_turn')
        return text

    app._call_anthropic = fake_call
    app._ddg_search = lambda query, max_results=4: []


# ----------------------------------------------------------------------------- מדדים
def fleiss_kappa(ratings):
    """ratings: רשימה של רשימות (לכל זוג, הסימנים של כל הסוכנים). Fleiss' kappa על ארבע הקטגוריות V/A/X/O."""
    cats = ['V', 'A', 'X', 'O']
    rows = [r for r in ratings if len(r) >= 2]
    if not rows:
        return None
    n = len(rows[0])
    rows = [r for r in rows if len(r) == n]
    N = len(rows)
    p_j = [sum(r.count(c) for r in rows) / (N * n) for c in cats]
    P_i = [(sum(r.count(c) ** 2 for c in cats) - n) / (n * (n - 1)) for r in rows]
    P_bar, P_e = sum(P_i) / N, sum(p * p for p in p_j)
    return None if P_e == 1 else (P_bar - P_e) / (1 - P_e)


def percentile(values, q):
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def offline_benchmark(app):
    """זמן חישוב הניתוח המלא (ללא AI) לפי כמות גורמים - 5 משיבים, צפיפות 25%."""
    import pandas as pd
    from ism_core import transitive_closure, compute_ism_levels, build_analysis_results, classify_micmac
    rows = []
    for n in (10, 20, 30, 40, 60):
        rnd = random.Random(n)
        F = [f'F{k}' for k in range(n)]
        experts = {f'E{e}': {'name': f'E{e}', 'role': 'r', 'responses': {
            F[i]: {F[j]: (rnd.choice('VAX') if rnd.random() < 0.25 else 'O') for j in range(i + 1, n)} for i in range(n)}}
            for e in range(5)}
        app.st.session_state.update({'FACTORS': F, 'EXPERT_DATA': experts, 'JUSTIFICATIONS': {}, 'AI_LOG': []})
        t0 = time.perf_counter()
        ssim, conflicts, *_ = app.calculate_ism_matrices()
        # הכרעת קונפליקטים לפי הרוב, בלי AI - מודדים רק את מנוע החישוב
        for c in conflicts:
            ssim.loc[c['pair'][0], c['pair'][1]] = c['counts'].most_common(1)[0][0]
        _, irm, frm, micmac = app.solve_conflicts_and_finalize(ssim, [])
        res = build_analysis_results(irm, frm, micmac)
        levels = compute_ism_levels(frm)
        rows.append({'factors': n, 'pairs': n * (n - 1) // 2, 'seconds': round(time.perf_counter() - t0, 3),
                     'levels': len(levels), 'saturation': round(res['saturation'], 3)})
    return rows


# ----------------------------------------------------------------------------- הריצה
def main():
    args = parse_args()
    secrets = load_secrets()
    api_key = secrets.get('ANTHROPIC_API_KEY') or os.environ.get('ANTHROPIC_API_KEY', '')
    if not args.dry_run and not api_key:
        sys.exit('לא נמצא ANTHROPIC_API_KEY ב-.streamlit/secrets.toml. הדבק את המפתח בשדה הייעודי והרץ שוב.')

    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ('_dry' if args.dry_run else '')
    out_dir = os.path.join(args.out, stamp)
    os.makedirs(out_dir, exist_ok=True)
    os.environ['ISM_DB_PATH'] = os.path.join(out_dir, 'sim.db')

    with contextlib.redirect_stderr(io.StringIO()):
        import project2025 as app
        import storage
    app.DB_PATH = os.path.join(out_dir, 'sim.db')  # מסד נפרד לריצה, גם אם secrets מגדיר מסד אחר
    storage.init_db(app.DB_PATH)
    app.st.session_state = {}
    app.init_session_state()
    app.st.session_state.update({'AI_PROVIDER': 'anthropic', 'ANTHROPIC_API_KEY': api_key or 'dry-run',
                                 'ANTHROPIC_WORKSPACE_ID': secrets.get('ANTHROPIC_WORKSPACE_ID', ''), 'TOPIC': args.topic})
    if args.dry_run:
        install_fake_ai(app)

    # מכשור: כשלי פענוח JSON וחיפושים
    stats = {'json_parse_failures': 0, 'searches': 0, 'search_results': 0, 'search_seconds': 0.0, 'errors': []}
    orig_parse, orig_search = app._try_parse_json, app._ddg_search

    def counted_parse(*a, **k):
        data, err = orig_parse(*a, **k)
        if err is not None:
            stats['json_parse_failures'] += 1
        return data, err

    def timed_search(*a, **k):
        t0 = time.perf_counter()
        res = orig_search(*a, **k)
        stats['searches'] += 1
        stats['search_results'] += len(res)
        stats['search_seconds'] += time.perf_counter() - t0
        return res

    app._try_parse_json, app._ddg_search = counted_parse, timed_search

    stages = []

    def stage(name):
        class S:
            def __enter__(self):
                self.t0 = time.perf_counter()
                self.calls0 = app.st.session_state['AI_USAGE']['calls']
                self.cost0 = app.st.session_state['AI_USAGE']['cost']
                print(f'▶ {name} ...', flush=True)

            def __exit__(self, et, ev, tb):
                u = app.st.session_state['AI_USAGE']
                rec = {'stage': name, 'seconds': round(time.perf_counter() - self.t0, 1),
                       'ai_calls': u['calls'] - self.calls0, 'cost_usd': round(u['cost'] - self.cost0, 4),
                       'ok': et is None}
                if et is not None:
                    rec['error'] = f'{et.__name__}: {ev}'
                    stats['errors'].append(f'{name}: {rec["error"]}')
                stages.append(rec)
                print(f'  ✓ {name}: {rec["seconds"]}s, {rec["ai_calls"]} קריאות, ${rec["cost_usd"]}' if et is None
                      else f'  ✗ {name}: {rec["error"]}', flush=True)
                return False
        return S()

    total_agents = args.expert_types * args.per_type
    print(f'מודל: {app.ANTHROPIC_MODEL_NAME} (effort {app.ANTHROPIC_EFFORT}) | נושא: {args.topic}')
    print(f'{args.factors} גורמים, {args.expert_types} סוגי מומחים × {args.per_type} = {total_agents} סוכנים')
    print(f'הערכת עלות גסה: ${0.03 + total_agents * 0.07 + 0.2:.2f}-${0.1 + total_agents * 0.15 + 0.5:.2f}\n', flush=True)
    t_start = time.perf_counter()

    # 1. גורמים
    with stage('חקר נושא והמלצה על גורמים'):
        recs, factor_cites = app._research_factors(args.topic, args.factors)
    factors = [r['name'] for r in recs][:args.factors]
    if len(factors) < 3:
        sys.exit(f'חקר הנושא החזיר רק {len(factors)} גורמים - עוצר.')
    app.st.session_state['FACTORS'] = factors
    storage.save_settings({'FACTORS': factors, 'TOPIC': args.topic, 'GENERIC_QUESTION': app.DEFAULT_GENERIC_QUESTION}, app.DB_PATH)

    # 2. סוגי מומחים
    with stage('המלצה על סוגי מומחים'):
        roles = app._get_ai_expert_recommendations(args.topic, factors, args.expert_types)
    roles = [r for r in (roles or []) if isinstance(r, dict) and r.get('role')][:args.expert_types]

    # 3. סוכנים
    agent_runs = []
    with stage(f'{len(roles) * args.per_type} סוכני מומחה סינתטיים'):
        for role in roles:
            for k in range(1, args.per_type + 1):
                label = f'#{k}' if args.per_type > 1 else None
                calls0 = app.st.session_state['AI_USAGE']['calls']
                fails0 = stats['json_parse_failures']
                t0 = time.perf_counter()
                try:
                    matrix, cites = app._simulate_synthetic_expert(role, factors)
                    app._inject_synthetic_data_or(role, matrix, factors, cites, instance_label=label)
                    answered = sum(len(v) for v in matrix.values())
                    ok, err = True, None
                except Exception as e:
                    answered, ok, err = 0, False, str(e)
                    stats['errors'].append(f'סוכן {role["role"]} {label or ""}: {e}')
                agent_runs.append({'role': role['role'], 'instance': k, 'ok': ok, 'error': err,
                                   'seconds': round(time.perf_counter() - t0, 1),
                                   'ai_calls': app.st.session_state['AI_USAGE']['calls'] - calls0,
                                   'json_retries': stats['json_parse_failures'] - fails0,
                                   'pairs_answered': answered, 'pairs_expected': len(factors) * (len(factors) - 1) // 2})
                print(f'    {"✓" if ok else "✗"} {role["role"]} {label or ""}: {agent_runs[-1]["seconds"]}s, '
                      f'{answered}/{agent_runs[-1]["pairs_expected"]} זוגות', flush=True)

    app.sync_from_db()
    n_experts = len(app.st.session_state['EXPERT_DATA'])
    if n_experts == 0:
        sys.exit('אף סוכן לא הצליח - עוצר. ראה שגיאות למעלה.')

    # 4. אגרגציה
    with stage('אגרגציה וזיהוי קונפליקטים'):
        ssim_init, conflicts, *_ = app.calculate_ism_matrices()
        non_o, n_conf, n_total = app.compute_ssim_density(ssim_init, factors)

    # 5. הכרעה וניתוח
    with stage(f'שופט AI ל-{len(conflicts)} קונפליקטים + חישוב הניתוח'):
        app.analyze_and_store(ssim_init, conflicts)
    results = app.st.session_state['RESULTS']

    # 6. דוח Excel של האפליקציה
    with stage('דוח Excel'):
        with open(os.path.join(out_dir, 'ISM_Report.xlsx'), 'wb') as f:
            f.write(app.build_excel_report())

    total_seconds = time.perf_counter() - t_start

    # ---------------------------------------------------------------- איסוף מדדים
    from ism_core import compute_ism_levels, ism_cycle_members
    frm = results['frm']
    levels = compute_ism_levels(frm)
    cycles = ism_cycle_members(frm)
    sym_stats = app.compute_expert_symbol_stats(factors).drop(columns=['_non_o_pct']).to_dict('records')
    pair_ratings = []
    for i, j in itertools.combinations(range(len(factors)), 2):
        fi, fj = factors[i], factors[j]
        pair_ratings.append([e['responses'].get(fi, {}).get(fj, 'O') for e in app.st.session_state['EXPERT_DATA'].values()])
    unanimous = sum(1 for r in pair_ratings if len(set(r)) == 1)
    judge = app.st.session_state.get('AI_LOG', [])
    judge_fallback = sum(1 for j in judge if 'שגיאת AI' in j.get('נימוק ה-AI', ''))
    history = app.st.session_state['AI_USAGE']['history']
    lat = [h['שניות'] for h in history if h.get('שניות') is not None]
    usage = app.st.session_state['AI_USAGE']

    def micmac_rows(key):
        m = results[key].sort_values('DP', ascending=False)
        return [{'factor': f, 'DP': float(r.DP), 'DEP': float(r.DEP), 'class': r.Classification} for f, r in m.iterrows()]

    report = {
        'run': stamp, 'dry_run': args.dry_run, 'model': app.ANTHROPIC_MODEL_NAME, 'effort': app.ANTHROPIC_EFFORT,
        'topic': args.topic, 'factors': factors, 'factor_sources': factor_cites,
        'expert_roles': roles, 'agents': agent_runs, 'stages': stages,
        'totals': {'seconds': round(total_seconds, 1), 'ai_calls': usage['calls'],
                   'input_tokens': usage['input_tokens'], 'output_tokens': usage['output_tokens'],
                   'cost_usd': round(usage['cost'], 4)},
        'latency_seconds': {'p50': percentile(lat, 0.5), 'p90': percentile(lat, 0.9), 'max': max(lat) if lat else None},
        'models_used': dict(Counter(h.get('מודל') for h in history)),
        'stop_reasons': dict(Counter(h.get('סיום') for h in history)),
        'reliability': {'agents_ok': sum(a['ok'] for a in agent_runs), 'agents_total': len(agent_runs),
                        'pair_coverage': round(sum(a['pairs_answered'] for a in agent_runs) /
                                               max(1, sum(a['pairs_expected'] for a in agent_runs)), 3),
                        'json_parse_failures': stats['json_parse_failures'],
                        'judge_fell_back_to_majority': judge_fallback, 'errors': stats['errors']},
        'search': {'calls': stats['searches'], 'results': stats['search_results'],
                   'seconds': round(stats['search_seconds'], 1)},
        'quality': {'experts': n_experts, 'pairs': n_total, 'direct_links': non_o,
                    'density_pct': round(non_o / n_total * 100, 1) if n_total else 0,
                    'conflicts': len(conflicts), 'unanimous_pairs': unanimous,
                    'fleiss_kappa': None if (k := fleiss_kappa(pair_ratings)) is None else round(k, 3),
                    'saturation': round(results['saturation'], 3), 'classic_degenerate': bool(results['classic_degenerate']),
                    'levels': levels, 'cycle_members': sorted(cycles),
                    'godet_power': results['godet_power'], 'katz_alpha': round(results['katz_alpha'], 4)},
        'expert_symbol_stats': sym_stats,
        'judge_decisions': judge,
        'micmac': {'classic': micmac_rows('micmac'), 'godet': micmac_rows('micmac_godet'), 'katz': micmac_rows('micmac_katz')},
        'calls': history,
    }
    with stage('בדיקת ביצועים של מנוע החישוב (ללא AI)'):
        report['offline_benchmark'] = offline_benchmark(app)

    with open(os.path.join(out_dir, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    write_markdown(report, os.path.join(out_dir, 'report.md'))
    print(f'\nהסתיים. {report["totals"]["ai_calls"]} קריאות, ${report["totals"]["cost_usd"]}, {report["totals"]["seconds"]}s')
    print(f'דוחות: {out_dir}')


def write_markdown(r, path):
    t, q, rel = r['totals'], r['quality'], r['reliability']
    L = [f"# דוח סימולציה: {r['topic']}", '',
         f"מודל `{r['model']}` (effort {r['effort']}){' — **ריצה מדומה**' if r['dry_run'] else ''} · {r['run']}", '',
         '## סיכום', '',
         f"| מדד | ערך |", '|---|---|',
         f"| זמן כולל | {t['seconds']} שניות |",
         f"| קריאות AI | {t['ai_calls']} |",
         f"| טוקנים (קלט / פלט) | {t['input_tokens']:,} / {t['output_tokens']:,} |",
         f"| עלות | ${t['cost_usd']} |",
         f"| זמן תגובה לקריאה (חציון / 90% / מקסימום) | {r['latency_seconds']['p50']} / {r['latency_seconds']['p90']} / {r['latency_seconds']['max']} שניות |",
         f"| סוכנים שהצליחו | {rel['agents_ok']}/{rel['agents_total']} |",
         f"| כיסוי זוגות בתשובות הסוכנים | {rel['pair_coverage']:.0%} |",
         f"| כשלי פענוח JSON (תוקנו בניסיון חוזר) | {rel['json_parse_failures']} |",
         f"| הכרעות שופט שנפלו לכלל הרוב | {rel['judge_fell_back_to_majority']} |",
         f"| סיומי קריאה | {r['stop_reasons']} |", '',
         '## שלבים', '', '| שלב | שניות | קריאות AI | עלות | תקין |', '|---|---|---|---|---|']
    L += [f"| {s['stage']} | {s['seconds']} | {s['ai_calls']} | ${s['cost_usd']} | {'✓' if s['ok'] else '✗ ' + s.get('error', '')} |" for s in r['stages']]
    L += ['', '## סוכנים', '', '| תפקיד | # | שניות | זוגות | ניסיונות JSON חוזרים | תקין |', '|---|---|---|---|---|---|']
    L += [f"| {a['role']} | {a['instance']} | {a['seconds']} | {a['pairs_answered']}/{a['pairs_expected']} | {a['json_retries']} | {'✓' if a['ok'] else '✗ ' + (a['error'] or '')} |" for a in r['agents']]
    L += ['', '## איכות התוצאה', '',
          f"- גורמים: {', '.join(r['factors'])}",
          f"- צפיפות קשרים ישירים: {q['density_pct']}% ({q['direct_links']}/{q['pairs']} זוגות)",
          f"- קונפליקטים בין הסוכנים: {q['conflicts']}; זוגות עם הסכמה מלאה: {q['unanimous_pairs']}",
          f"- הסכמה בין הסוכנים (Fleiss' kappa): {q['fleiss_kappa']}",
          f"- רוויית FRM: {q['saturation']} ({'מנוון לסיווג הקלאסי' if q['classic_degenerate'] else 'תקין'})",
          f"- רמות ISM: {len(q['levels'])}; גורמים במעגל: {', '.join(q['cycle_members']) or 'אין'}",
          '', '### רמות (מהתלוי ביותר למניע ביותר)', '']
    L += [f"{k + 1}. {', '.join(lvl)}" for k, lvl in enumerate(q['levels'])]
    for key, title in (('katz', 'MICMAC מצטבר (Katz)'), ('classic', 'MICMAC קלאסי')):
        L += ['', f'### {title}', '', '| גורם | DP | DEP | סיווג |', '|---|---|---|---|']
        L += [f"| {m['factor']} | {m['DP']} | {m['DEP']} | {m['class']} |" for m in r['micmac'][key]]
    L += ['', '## ביצועי מנוע החישוב (ללא AI)', '', '| גורמים | זוגות | שניות | רמות | רוויה |', '|---|---|---|---|---|']
    L += [f"| {b['factors']} | {b['pairs']} | {b['seconds']} | {b['levels']} | {b['saturation']} |" for b in r['offline_benchmark']]
    if rel['errors']:
        L += ['', '## שגיאות', ''] + [f'- {e}' for e in rel['errors']]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
