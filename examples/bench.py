"""A small benchmark: five local form tasks of rising difficulty, run with each decision model.

    uv run python examples/bench.py run --label jev --decider jev --repeats 3
    uv run python examples/bench.py run --label llm --decider llm --model openrouter/deepseek/deepseek-v4.1-flash --repeats 3
    uv run python examples/bench.py report tmp/bench/*.jsonl

Every page posts what was submitted to a local server, so a run counts as correct only when the right values arrived,
whatever the agent claims. Each run uses a fresh run id, so several configurations can run side by side.
"""
import argparse
import functools
import glob
import http.server
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(HERE, "bench_site")

TASKS = [
    {
        "id": "t1",
        "name": "contact form",
        "page": "t1-contact.html",
        "goal": "Send the team a message through the contact form.",
        "facts": {"name": "Alex Morgan", "email": "alex.morgan@example.com", "message": "Please send the pricing sheet."},
        "expect": {"name": "Alex Morgan", "email": "alex.morgan@example.com", "message": "Please send the pricing sheet."},
    },
    {
        "id": "t2",
        "name": "demo request in a new tab",
        "page": "t2-home.html",
        "goal": "Request a demo: open the demo request page, fill in the form, tick the consent box and send it.",
        "facts": {"name": "Alex Morgan", "email": "alex.morgan@example.com", "team_size": "6 to 20", "message": "We want to see queues and storage."},
        "expect": {"name": "Alex Morgan", "email": "alex.morgan@example.com", "size": "6 to 20", "consent": True},
    },
    {
        "id": "t3",
        "name": "three step wizard",
        "page": "t3-wizard.html",
        "goal": "Create a workspace: go through all the steps, choose the Team plan, confirm and finish.",
        "facts": {"email": "alex.morgan@example.com", "name": "Alex Morgan", "plan": "Team"},
        "expect": {"email": "alex.morgan@example.com", "name": "Alex Morgan", "plan": "Team", "confirm": True},
    },
    {
        "id": "t4",
        "name": "long form with scrolling",
        "page": "t4-long.html",
        "goal": "Apply to the partner program with the facts and submit the application.",
        "facts": {"first_name": "Alex", "last_name": "Morgan", "email": "alex.morgan@example.com", "phone": "555 0100", "company": "Northwind Labs",
                  "role": "Engineer", "country": "Canada", "message": "We build developer tools and want to resell Acme Cloud."},
        "expect": {"first": "Alex", "last": "Morgan", "email": "alex.morgan@example.com", "phone": "555 0100", "company": "Northwind Labs",
                   "role": "Engineer", "country": "Canada", "consent": True},
    },
    {
        "id": "t5",
        "name": "sign up with a 40 country dropdown",
        "page": "t5-signup.html",
        "goal": "Create a personal account with the facts, accept the terms and finish.",
        "facts": {"email": "alex.morgan@example.com", "password": "Correct-Horse-9", "country": "Canada", "account_type": "Personal"},
        "expect": {"email": "alex.morgan@example.com", "password": "Correct-Horse-9", "again": "Correct-Horse-9", "country": "Canada", "kind": "Personal", "terms": True},
    },
]


class Handler(http.server.SimpleHTTPRequestHandler):
    submissions = {}

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        Handler.submissions[body.get("run", "")] = body
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')


def serve():
    handler = functools.partial(Handler, directory=SITE)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def same(expected, got):
    if isinstance(expected, bool):
        return got is expected
    return str(got).strip().lower() == str(expected).strip().lower()


def check(task, submission):
    """(correct, what was wrong)"""
    if not submission:
        return False, "nothing was submitted"
    values = submission.get("values", {})
    wrong = [key for key, expected in task["expect"].items() if not same(expected, values.get(key))]
    if wrong:
        return False, "wrong or missing: " + ", ".join(wrong)
    return True, ""


def command_run(args):
    server = serve()
    port = server.server_address[1]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    wanted = args.tasks.split(",") if args.tasks else [task["id"] for task in TASKS]
    for task in TASKS:
        if task["id"] not in wanted:
            continue
        for repeat in range(args.repeats):
            run_id = f"{args.label}-{task['id']}-{repeat + 1}-{int(time.time())}"
            url = f"http://127.0.0.1:{port}/{task['page']}?run={urllib.parse.quote(run_id)}"
            command = [sys.executable, "-m", "mmjb.cli", "run", "--goal", task["goal"], "--url", url, "--allow-writes",
                       "--max-steps", str(args.max_steps), "--budget", str(args.budget), "--json"]
            for key, value in task["facts"].items():
                command += ["--fact", f"{key}={value}"]
            environment = dict(os.environ, MMJB_DECIDER=args.decider)
            if args.model:
                environment["MMJB_DECIDER_MODEL"] = args.model
            started = time.time()
            done = subprocess.run(command, capture_output=True, text=True, env=environment)
            try:
                result = json.loads(done.stdout)
            except ValueError:
                result = {"outcome": "error", "steps": [], "seconds": time.time() - started, "cost": 0.0, "decider_calls": 0, "fallback_calls": 0,
                          "notes": [done.stderr[-300:]]}
            correct, why = check(task, Handler.submissions.get(run_id))
            row = {"label": args.label, "decider": args.decider, "model": args.model, "task": task["id"], "task_name": task["name"], "repeat": repeat + 1,
                   "outcome": result["outcome"], "correct": correct, "why": why, "steps": len(result["steps"]), "seconds": result["seconds"],
                   "cost": result["cost"], "decider_calls": result["decider_calls"], "fallback_calls": result["fallback_calls"],
                   "fallback_cost": result.get("fallback_cost", 0.0)}
            with open(args.out, "a") as handle:
                handle.write(json.dumps(row) + "\n")
            mark = "correct" if correct else "WRONG  "
            print(f"{args.label:<10} {task['id']} #{repeat + 1} {mark} {row['outcome']:<10} {row['steps']:>2} steps {row['seconds']:>6.1f}s ${row['cost']:.4f} {why}", file=sys.stderr)
    server.shutdown()


def money(value):
    if value < 0.0001:
        return f"${value:.6f}"
    if value < 0.01:
        return f"${value:.4f}"
    return f"${value:.2f}"


def command_report(args):
    rows = []
    for pattern in args.files:
        for path in sorted(glob.glob(pattern)):
            with open(path) as handle:
                for line in handle:
                    if line.strip():
                        rows.append(json.loads(line))
    labels = []
    for row in rows:
        if row["label"] not in labels:
            labels.append(row["label"])
    names = dict(item.split("=", 1) for item in args.names) if args.names else {}
    tasks = [task for task in TASKS if any(row["task"] == task["id"] for row in rows)]
    summary = {}
    for label in labels:
        mine = [row for row in rows if row["label"] == label]
        steps = sum(row["steps"] for row in mine)
        summary[label] = {
            "runs": len(mine),
            "correct": sum(1 for row in mine if row["correct"]),
            "claimed": sum(1 for row in mine if row["outcome"] == "verified"),
            "steps": statistics.median(row["steps"] for row in mine),
            "seconds": statistics.median(row["seconds"] for row in mine),
            "cost": statistics.median(row["cost"] for row in mine),
            "total_cost": sum(row["cost"] for row in mine),
            "total_steps": steps,
            "per_step": sum(row["cost"] for row in mine) / max(steps, 1),
            "seconds_per_step": sum(row["seconds"] for row in mine) / max(steps, 1),
            "fallback_runs": sum(1 for row in mine if row["fallback_calls"] > 0),
        }
    title = lambda label: names.get(label, label)
    print("### Accuracy, speed and cost on the benchmark\n")
    print("| decision model | correct submissions | agent said verified | median steps | median seconds | median cost per task | runs that needed the LLM fallback |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for label in labels:
        s = summary[label]
        print(f"| {title(label)} | **{s['correct']} of {s['runs']}** ({100 * s['correct'] // s['runs']}%) | {s['claimed']} of {s['runs']} | {s['steps']:.0f} | {s['seconds']:.1f} | {money(s['cost'])} | {s['fallback_runs']} of {s['runs']} |")
    print("\n### Correct submissions by task\n")
    print("| task | " + " | ".join(title(label) for label in labels) + " |")
    print("| --- | " + " | ".join("---" for _ in labels) + " |")
    for task in tasks:
        cells = []
        for label in labels:
            mine = [row for row in rows if row["label"] == label and row["task"] == task["id"]]
            cells.append(f"{sum(1 for row in mine if row['correct'])} of {len(mine)}" if mine else "n/a")
        print(f"| {task['name']} | " + " | ".join(cells) + " |")
    print("\n### Median steps and seconds by task\n")
    print("| task | " + " | ".join(title(label) for label in labels) + " |")
    print("| --- | " + " | ".join("---" for _ in labels) + " |")
    for task in tasks:
        cells = []
        for label in labels:
            mine = [row for row in rows if row["label"] == label and row["task"] == task["id"]]
            if mine:
                cells.append(f"{statistics.median(r['steps'] for r in mine):.0f} steps, {statistics.median(r['seconds'] for r in mine):.0f} s, {money(statistics.median(r['cost'] for r in mine))}")
            else:
                cells.append("n/a")
        print(f"| {task['name']} | " + " | ".join(cells) + " |")
    print("\n### What one step costs\n")
    print("| decision model | cost per step | seconds per step | steps per dollar |")
    print("| --- | --- | --- | --- |")
    for label in labels:
        s = summary[label]
        per_dollar = f"{1 / s['per_step']:,.0f}" if s["per_step"] > 0 else "n/a"
        print(f"| {title(label)} | {money(s['per_step'])} | {s['seconds_per_step']:.1f} | {per_dollar} |")
    print("\n### The same models on longer tasks (measured cost per step, multiplied)\n")
    print("A cost per step measured above, times the number of steps. Real long tasks also hit more unsure steps, so treat the larger columns as a floor for the cheaper models.\n")
    sizes = [("one form (8 steps)", 8), ("signup flow (25 steps)", 25), ("complex workflow (60 steps)", 60), ("1,000 forms (8 steps each)", 8000), ("10,000 workflow steps", 10000)]
    print("| decision model | " + " | ".join(name for name, _ in sizes) + " |")
    print("| --- | " + " | ".join("---" for _ in sizes) + " |")
    for label in labels:
        s = summary[label]
        print(f"| {title(label)} | " + " | ".join(money(s["per_step"] * count) for _, count in sizes) + " |")
    print("\n### Time on longer tasks\n")
    print("| decision model | one form (8 steps) | signup flow (25 steps) | complex workflow (60 steps) |")
    print("| --- | --- | --- | --- |")
    for label in labels:
        s = summary[label]
        print(f"| {title(label)} | {s['seconds_per_step'] * 8:.0f} s | {s['seconds_per_step'] * 25 / 60:.1f} min | {s['seconds_per_step'] * 60 / 60:.1f} min |")
    base = labels[-1] if args.baseline not in labels else args.baseline
    print(f"\n### Cost compared with {title(base)}\n")
    print("| decision model | cost per step compared with the baseline |")
    print("| --- | --- |")
    for label in labels:
        s = summary[label]
        ratio = summary[base]["per_step"] / s["per_step"] if s["per_step"] > 0 else 0
        print(f"| {title(label)} | {ratio:.1f}x cheaper |" if ratio >= 1.05 else f"| {title(label)} | {1 / ratio:.1f}x more expensive |" if ratio > 0 else f"| {title(label)} | n/a |")
    wrong = [row for row in rows if not row["correct"]]
    if wrong:
        print("\n### Runs that did not end with the right submission\n")
        print("| decision model | task | run | outcome | what was wrong |")
        print("| --- | --- | --- | --- | --- |")
        for row in wrong:
            print(f"| {title(row['label'])} | {row['task_name']} | {row['repeat']} | {row['outcome']} | {row['why']} |")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--label", required=True)
    run.add_argument("--decider", required=True)
    run.add_argument("--model", default="")
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--tasks", default="")
    run.add_argument("--max-steps", type=int, default=30)
    run.add_argument("--budget", type=int, default=240)
    run.add_argument("--out", default="")
    report = commands.add_parser("report")
    report.add_argument("files", nargs="+")
    report.add_argument("--names", nargs="*", default=[], help="label=Display name")
    report.add_argument("--baseline", default="")
    args = parser.parse_args()
    if args.command == "run":
        args.out = args.out or os.path.join("tmp", "bench", f"{args.label}.jsonl")
        command_run(args)
    else:
        command_report(args)


if __name__ == "__main__":
    main()
