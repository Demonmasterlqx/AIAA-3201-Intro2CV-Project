from pathlib import Path
from datetime import datetime, timezone
import csv, json, re, subprocess

ROOT = Path('/data/home/sim6g/code/aiaa3201_cv_project/ascent')
REPORT = ROOT / 'ascent_qwenvl_objectpatch' / 'report' / 'part3_qwenvl_report'
REPORT.mkdir(parents=True, exist_ok=True)

RUNS = [
    {
        'id': 'single_20scene_deepseek',
        'method': 'ASCENT-Single',
        'role': 'standard_sota_pipeline',
        'sample': 'matched_20_episode',
        'path': ROOT / 'ascent/results/baseline_compare_single/20260426_single_20scene_deepseek_compare',
        'notes': 'Part 2 standard single-agent ASCENT baseline on the matched 20-episode scene subset.'
    },
    {
        'id': 'naive_2agents_20scene_deepseek',
        'method': 'ASCENT-2Agents-Naive',
        'role': 'multi_agent_baseline',
        'sample': 'matched_20_episode',
        'path': ROOT / 'agents_ascent/results/baseline_compare_naive/20260426_naive_20scene_deepseek_compare',
        'notes': 'Naive always-on two-agent baseline; useful negative control for coordination without cognitive final confirmation.'
    },
    {
        'id': 'qwenvl_fullframe_20scene',
        'method': 'Qwen-VL Full-frame Final Check',
        'role': 'part3_ablation_fullframe',
        'sample': 'matched_20_episode',
        'path': ROOT / 'ascent_qwenvl_fullframe/results/baseline_compare_single/qwenvl_fullframe_20scene',
        'notes': 'Final BLIP2 double-check replaced by Qwen-VL on the whole egocentric frame.'
    },
    {
        'id': 'qwenvl_objectpatch_weakcache_20scene',
        'method': 'Qwen-VL Object-patch Final Check',
        'role': 'part3_enhanced_pipeline_headline',
        'sample': 'matched_20_episode',
        'path': ROOT / 'ascent_qwenvl_objectpatch/results/baseline_compare_single/qwenvl_objectpatch_qwen_weakcache_20eps_v1',
        'notes': 'Best current matched-20 result before commit 08ada94; object crops, cache reuse, and weak positive cache support.'
    },
    {
        'id': 'qwenvl_objectpatch_1000_early',
        'method': 'Qwen-VL Object-patch Early 1000eps',
        'role': 'large_sample_risk_evidence',
        'sample': '1000_episode_early_version',
        'path': ROOT / 'ascent_qwenvl_objectpatch/results/baseline_compare_single/qwenvl_objectpatch_1000eps_v1',
        'notes': 'Early objectpatch implementation; not headline final method, used only to discuss scaling instability and failure modes.'
    },
    {
        'id': 'qwenvl_objectpatch_cached_20scene',
        'method': 'Object-patch + verified target cache',
        'role': 'tuned_ablation',
        'sample': 'matched_20_episode',
        'path': ROOT / 'ascent_qwenvl_objectpatch/results/baseline_compare_single/qwenvl_objectpatch_qwen_approach_cached_20eps_v1',
        'notes': 'Ablation immediately preceding weak-cache support.'
    },
    {
        'id': 'qwenvl_frontier_radius_stop_20scene',
        'method': 'Object-patch + frontier radius/stop',
        'role': 'failed_or_mixed_ablation',
        'sample': 'matched_20_episode',
        'path': ROOT / 'ascent_qwenvl_objectpatch/results/baseline_compare_single/qwenvl_objectpatch_frontier_radius_stop_20eps_v1',
        'notes': 'SPL-oriented tuning attempt that did not become the final report method.'
    },
    {
        'id': 'qwenvl_verified_cloud_stick_20scene',
        'method': 'Object-patch + verified cloud stick-frontier',
        'role': 'failed_or_mixed_ablation',
        'sample': 'matched_20_episode',
        'path': ROOT / 'ascent_qwenvl_objectpatch/results/baseline_compare_single/qwenvl_objectpatch_verified_cloud_stick_20eps_v1',
        'notes': 'Exploration-stickiness tuning attempt; retained as ablation evidence.'
    },
    {
        'id': 'qwenvl_conservative_stair_8eps',
        'method': 'Object-patch + conservative stair waypoint',
        'role': 'failed_or_mixed_ablation',
        'sample': '8_episode_ablation',
        'path': ROOT / 'ascent_qwenvl_objectpatch/results/baseline_compare_single/qwenvl_objectpatch_conservative_stair_8eps_v1',
        'notes': 'Recent 8-episode tuning attempt; improved SPL on small subset but reduced SR versus weakcache8.'
    },
    {
        'id': 'qwenvl_current08ada94_8eps',
        'method': 'Object-patch + multi-view weak confirmation (08ada94)',
        'role': 'rejected_current_commit_ablation',
        'sample': '8_episode_ablation',
        'path': ROOT / 'ascent_qwenvl_objectpatch/results/baseline_compare_single/qwenvl_objectpatch_current08ada94_8eps_v1',
        'notes': 'Mandatory current-commit rerun. Rejected because SR fell to 75% and introduced a false-positive failure; therefore no 20-episode expansion was launched.'
    },
    {
        'id': 'qwenvl_timeoutguard_8eps',
        'method': 'Object-patch + timeout guard',
        'role': 'failed_ablation',
        'sample': '8_episode_ablation',
        'path': ROOT / 'ascent_qwenvl_objectpatch/results/baseline_compare_single/qwenvl_objectpatch_qwen_timeoutguard_8eps_v1',
        'notes': 'Rejected because it introduced false positives and reduced SR.'
    },
    {
        'id': 'qwenvl_tvambig_8eps',
        'method': 'Object-patch + TV ambiguity tightening',
        'role': 'failed_ablation',
        'sample': '8_episode_ablation',
        'path': ROOT / 'ascent_qwenvl_objectpatch/results/baseline_compare_single/qwenvl_objectpatch_tvambig_8eps_v1',
        'notes': 'Rejected because tighter ambiguity handling reduced SR.'
    },
]

avg_re = re.compile(r'Average episode ([A-Za-z0-9_]+):\s*([-+0-9.]+)')
till_re = re.compile(r'Till Now Average Success rate:\s*([0-9.]+)% \((\d+) out of (\d+)\)')
fail_re = re.compile(r"failed due to '([^']+)'")
scene_fail_re = re.compile(r"Episode\s+([^\s]+) in scene ([^\s]+) failed due to '([^']+)'")
scene_sr_re = re.compile(r'Success rate of Scene .*?/([^/]+)\.basis\.glb:\s*([0-9.]+)% \(([0-9.]+) out of ([0-9]+)\)')
exit_re = re.compile(r'EXIT_CODE:(\d+)')

current_commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT/'ascent_qwenvl_objectpatch', text=True).strip()
current_branch = subprocess.check_output(['git','rev-parse','--abbrev-ref','HEAD'], cwd=ROOT/'ascent_qwenvl_objectpatch', text=True).strip()

summaries = []
failure_rows = []
scene_rows = []
for spec in RUNS:
    d = spec['path']
    log = d / 'run.log'
    row = {k: str(v) if isinstance(v, Path) else v for k,v in spec.items() if k != 'path'}
    row['result_dir'] = str(d)
    row['run_log'] = str(log)
    row['exists'] = log.exists()
    row['commit'] = None
    cfile = d / 'commit.txt'
    if cfile.exists():
        row['commit'] = cfile.read_text(errors='ignore').strip().splitlines()[0]
    if not log.exists():
        summaries.append(row)
        continue
    txt = log.read_text(errors='ignore')
    lines = txt.splitlines()
    avgs = {}
    evidence = []
    for line in lines:
        m = avg_re.search(line)
        if m:
            avgs[m.group(1)] = float(m.group(2))
            evidence.append(line)
    tills = till_re.findall(txt)
    if tills:
        row['till_now_sr_percent'] = float(tills[-1][0])
        row['completed_successes'] = int(tills[-1][1])
        row['completed_episodes'] = int(tills[-1][2])
    for key, val in avgs.items():
        row['avg_' + key] = val
    row['success_percent'] = round(100.0 * avgs['success'], 4) if 'success' in avgs else row.get('till_now_sr_percent')
    row['spl_percent'] = round(100.0 * avgs['spl'], 4) if 'spl' in avgs else None
    row['failure_count'] = len(fail_re.findall(txt))
    row['qwen_success_lines'] = txt.count('Qwen-VL patch double check success') + txt.count('Qwen-VL double check success')
    row['blip_success_lines'] = txt.count('Double check success!!!') - txt.count('Qwen-VL double check success') - txt.count('Qwen-VL patch double check success')
    row['stalled_frontier_events'] = txt.count('Disabled stalled exploration frontier')
    row['climb_stair_success_events'] = txt.count('climb stair success')
    exits = exit_re.findall(txt)
    row['exit_code'] = exits[-1] if exits else None
    fail_counts = {}
    for cause in fail_re.findall(txt):
        fail_counts[cause] = fail_counts.get(cause, 0) + 1
    row['failure_buckets'] = fail_counts
    for ep, scene, cause in scene_fail_re.findall(txt):
        failure_rows.append({'run_id': spec['id'], 'method': spec['method'], 'episode_id': ep, 'scene_id': scene, 'failure_cause': cause})
    for scene, sr, succ, total in scene_sr_re.findall(txt):
        scene_rows.append({'run_id': spec['id'], 'method': spec['method'], 'scene_id': scene, 'scene_success_rate_percent': float(sr), 'scene_successes': float(succ), 'scene_trials': int(total)})
    row['evidence_tail'] = evidence[-24:]
    summaries.append(row)

summary = {
    'generated_at_utc': datetime.now(timezone.utc).isoformat(),
    'source_worktree': str(ROOT / 'ascent_qwenvl_objectpatch'),
    'current_branch': current_branch,
    'current_commit': current_commit,
    'runs': summaries,
}
(REPORT / 'metrics_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')

csv_fields = ['id','method','role','sample','success_percent','spl_percent','avg_distance_to_goal','avg_num_steps','avg_final_check_call_count','avg_final_check_success_count','avg_final_check_parse_error_count','avg_final_check_confidence','avg_target_detected','avg_stop_called','failure_count','completed_successes','completed_episodes','qwen_success_lines','blip_success_lines','stalled_frontier_events','climb_stair_success_events','commit','run_log','notes']
with (REPORT / 'metrics_summary.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=csv_fields)
    w.writeheader()
    for r in summaries:
        w.writerow({k: r.get(k) for k in csv_fields})

with (REPORT / 'failure_taxonomy.csv').open('w', newline='', encoding='utf-8') as f:
    fields = ['run_id','method','failure_cause','count']
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in summaries:
        for cause, count in (r.get('failure_buckets') or {}).items():
            w.writerow({'run_id': r['id'], 'method': r['method'], 'failure_cause': cause, 'count': count})

with (REPORT / 'episode_failures.csv').open('w', newline='', encoding='utf-8') as f:
    fields = ['run_id','method','episode_id','scene_id','failure_cause']
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(failure_rows)

with (REPORT / 'scene_success_rates.csv').open('w', newline='', encoding='utf-8') as f:
    fields = ['run_id','method','scene_id','scene_success_rate_percent','scene_successes','scene_trials']
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(scene_rows)

print('WROTE', REPORT)
for r in summaries:
    print(r['id'], r.get('success_percent'), r.get('spl_percent'), r.get('avg_final_check_call_count'), r.get('completed_episodes'), r.get('commit'))
