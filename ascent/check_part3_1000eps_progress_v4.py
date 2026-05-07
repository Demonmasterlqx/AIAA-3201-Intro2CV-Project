from pathlib import Path
import collections
import re
import subprocess


base = Path("/data/home/sim6g/code/aiaa3201_cv_project/ascent")
queues = {
    "queue_v2": base / "part3_1000eps_queue_v2.log",
    "objectpatch_v3_queue": base / "part3_1000eps_objectpatch_v3.log",
    "objectpatch_v4_queue": base / "part3_1000eps_objectpatch_v4.log",
}
logs = {
    "single_v2": base
    / "ascent/results/baseline_compare_single/20260502_single_1000eps_rerun_v2/run.log",
    "objectpatch_v2_failed": base
    / "ascent_qwenvl_objectpatch_weakcache_rerun/results/baseline_compare_single/qwenvl_objectpatch_weakcache_1000eps_rerun_v2/run.log",
    "objectpatch_v3_failed": base
    / "ascent_qwenvl_objectpatch_weakcache_rerun/results/baseline_compare_single/qwenvl_objectpatch_weakcache_1000eps_rerun_v3/run.log",
    "objectpatch_v4": base
    / "ascent_qwenvl_objectpatch_weakcache_rerun/results/baseline_compare_single/qwenvl_objectpatch_weakcache_1000eps_rerun_v4/run.log",
}


def print_section(title: str) -> None:
    print(f"\n### {title}")


for name, queue in queues.items():
    print_section(name)
    print("exists", queue.exists(), queue)
    if queue.exists():
        print(queue.read_text(errors="ignore")[-3000:])

print_section("tmux")
try:
    out = subprocess.check_output(["tmux", "ls"], text=True, stderr=subprocess.STDOUT)
except Exception as exc:
    out = str(exc)
for line in out.splitlines():
    if (
        "part3_1000eps_rerun_v2" in line
        or "part3_objectpatch_1000eps_v3" in line
        or "part3_objectpatch_1000eps_v4" in line
        or "qwenvl_14187" in line
    ):
        print(line)

print_section("processes")
try:
    ps = subprocess.check_output(["ps", "-eo", "pid,etime,pcpu,pmem,cmd"], text=True)
    for line in ps.splitlines():
        if (
            "part3_1000eps" in line
            or "part3_objectpatch_1000eps_v" in line
            or "ascent.run" in line
            or "qwen25_vl" in line
        ) and "grep" not in line:
            print(line)
except Exception as exc:
    print("PS_ERR", exc)

for name, path in logs.items():
    print_section(name)
    print("exists", path.exists(), path)
    if not path.exists():
        continue
    text = path.read_text(errors="ignore")
    matches = list(
        re.finditer(
            r"Till Now Average Success rate:\s*([0-9.]+)% \((\d+) out of (\d+)\)",
            text,
        )
    )
    if matches:
        latest = matches[-1]
        print("completed", latest.group(3))
        print("successes", latest.group(2))
        print("sr_percent", latest.group(1))
    else:
        print("completed", 0)
    spl = re.findall(r"Till Now Average Spl:\s*([0-9.]+)%", text)
    dtg = re.findall(r"Till Now Average Dtg:\s*([0-9.eE+-]+)", text)
    print("latest_spl_percent", spl[-1] if spl else None)
    print("latest_dtg", dtg[-1] if dtg else None)
    avgs = re.findall(r"Average episode ([A-Za-z0-9_]+):\s*([-+0-9.]+)", text)
    if avgs:
        print("final_avg_tail", avgs[-16:])
    failures = collections.Counter(re.findall(r"failed due to '([^']+)'", text))
    print("failure_buckets", dict(failures.most_common()))
    print("qwen_patch_success", text.count("Qwen-VL patch double check success"))
    print("qwen_patch_final_checks", text.count("Qwen-VL patch final check"))
    print("qwen_patch_rejects", text.count("Qwen-VL patch double check rejected"))
    exits = re.findall(r"EXIT_CODE:(\d+)", text)
    print("exit", exits[-1] if exits else None)
