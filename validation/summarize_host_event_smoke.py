"""Offline correctness/ownership accounting for run_host_event_smoke.sh."""
import argparse
import collections
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--clients', type=int, default=16)
    args = parser.parse_args()
    root = args.run_dir
    rows = [row for f in root.glob('multiturn-*.json')
            for row in json.loads(f.read_text())]
    prefill = {}
    for f in root.glob('raw/prefill-*/*.log'):
        for line in f.read_text().splitlines():
            item = json.loads(line)
            if item['event'] == 'request.finished':
                prefill[item['rid']] = item['out']['meta_info']
    event_ids = collections.defaultdict(set)
    timing = collections.defaultdict(list)
    refill_engines = []
    for f in root.glob('logs/*.log'):
        if not f.name.startswith(('prefill-', 'decode-')):
            continue
        contents = f.read_text()
        if 'AgenticKV host_event_refill_enabled ' in contents:
            refill_engines.append(f.stem)
        for event, sid in re.findall(r'AgenticKV (\S+) snapshot=(\S+)', contents):
            event_ids[event].add(sid)
        for line in contents.splitlines():
            if 'AgenticKV h2d_stage_timing ' in line:
                for key in ('selected_to_grant_ms', 'io_to_fence_ms', 'fence_to_handoff_ms'):
                    match = re.search(rf'{key}=([\d.]+)', line)
                    if match:
                        timing[key].append(float(match[1]))
    expected = {f"{r['request_id']}:{r['generation']-1}" for r in rows if r['generation']}
    direct = event_ids['early_direct_bind'] & expected
    host = event_ids['shared_host_h2d_release'] & expected
    cache_rows = []
    for g in range(3):
        group = [r for r in rows if r['generation'] == g]
        cache_rows.append({
            'generation': g, 'n': len(group),
            'prompt_cached_pairs': dict(collections.Counter(
                f"{len(r['prompt_ids'])}/{prefill[r['response']['meta_info']['id']]['cached_tokens']}"
                for r in group)),
        })
    by_generation = {(r['request_id'], r['generation']): r for r in rows}
    cache_ok = all(
        prefill[r['response']['meta_info']['id']]['cached_tokens']
        == (len(by_generation[(r['request_id'], r['generation']-1)]['prompt_ids']) // 64) * 64
        for r in rows if r['generation']
    )
    ledger_states = {}
    for name in ('host.json', 'p2d-host.json'):
        obj = json.loads((root / 'control-before-stop' / name).read_text())
        entries = {}
        for f in (root / 'control-before-stop' / (name + '.events')).glob('*.json'):
            event = json.loads(f.read_text())
            if event.get('entry') is not None:
                entries[event['snapshot_id']] = event['entry']
        entries.update(obj['entries'])
        ledger_states[name] = dict(collections.Counter(
            entry.get('state', 'unknown') for entry in entries.values()))
    conservation_events = ('host_staging_offer', 'shared_host_d2h_complete',
                           'd_release_after_p_host', 'shared_host_h2d_release',
                           'shared_host_final_release', 'p2d_host_d2h_queued')
    offers, durable, source_release = [event_ids[k] for k in conservation_events[:3]]
    summary = {
        'client_trajectories': len(rows),
        'exact_token_matches': sum(r.get('exact_output_match') is True for r in rows),
        'cross_rounds': len(expected), 'direct': len(direct), 'host': len(host),
        'missing_or_double_route': sorted((expected - direct - host) | (direct & host)),
        'prefix_at_expected_page_boundary': cache_ok,
        'prefill_prompt_cached_tokens': cache_rows,
        'refill_executed_on': sorted(refill_engines),
        'ownership_event_counts': {k: len(event_ids[k]) for k in conservation_events},
        'host_offer_durable_source_release_equal': offers == durable == source_release,
        'host_durable_without_consume_or_final': sorted(durable - event_ids['shared_host_h2d_release']
                                                       - event_ids['shared_host_final_release']),
        'ledger_states_before_stop': ledger_states,
        'ledger_nonterminal_states_before_stop': {
            name: {state: count for state, count in states.items()
                   if state not in {'rejected', 'consumed'}}
            for name, states in ledger_states.items()},
        'h2d_stage_ms': {k: {'n': len(v), 'mean': sum(v)/len(v), 'max': max(v)}
                         for k, v in timing.items()},
    }
    print(json.dumps(summary, indent=2))
    (root / 'smoke-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    assert len(rows) == args.clients * 6
    assert summary['exact_token_matches'] == len(rows)
    assert len(expected) == args.clients * 4
    assert not summary['missing_or_double_route'] and cache_ok
    assert offers == durable == source_release
    assert not summary['host_durable_without_consume_or_final']
    assert all(not states for states in summary['ledger_nonterminal_states_before_stop'].values())
    assert len(refill_engines) == 2


if __name__ == '__main__':
    main()
