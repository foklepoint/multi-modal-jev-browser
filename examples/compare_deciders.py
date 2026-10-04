"""Run the demo form with each decision model several times and print a markdown table.

    uv run python examples/compare_deciders.py --runs 3 --deciders jev,clef,llm --out tmp/compare

It serves examples/demo_site on a local port, runs `mmjb run --json --record` once per decider per repeat, and reports
the median seconds, the median cost, the steps and how often the LLM fallback was needed. Each decider needs its own keys
(see .env.example); a decider whose keys are missing is skipped.
"""
import argparse
import functools
import http.server
import json
import os
import statistics
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
GOAL = ("Request a demo of Acme Cloud: open the demo request page, fill in the name, work email and team size, "
        "tick the consent box and send the request.")
FACTS = ["name=Alex Morgan", "email=alex.morgan@example.com", "team_size=6 to 20", "message=We want to see queues and storage."]
NEEDS = {
    "clef": ["CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"],
    "jev": ["OPENROUTER_API_KEY"],
    "llm": ["OPENROUTER_API_KEY"],
}


def serve(port):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=os.path.join(HERE, "demo_site"))

    class Quiet(http.server.ThreadingHTTPServer):
        def log_message(self, *args):
            pass

    server = Quiet(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--deciders", default="jev,clef,llm")
    parser.add_argument("--out", default="tmp/compare")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--llm-model", default="openrouter/deepseek/deepseek-v4.1-flash")
    args = parser.parse_args()
    server = serve(args.port)
    url = f"http://127.0.0.1:{args.port}/index.html"
    rows = []
    for decider in args.deciders.split(","):
        missing = [name for name in NEEDS[decider] if not os.environ.get(name)]
        if missing:
            print(f"skipping {decider}: set {', '.join(missing)}", file=sys.stderr)
            continue
        results = []
        for repeat in range(args.runs):
            record = os.path.join(args.out, f"{decider}-{repeat + 1}")
            command = [sys.executable, "-m", "mmjb.cli", "run", "--goal", GOAL, "--url", url, "--allow-writes",
                       "--max-steps", "20", "--json", "--record", record]
            for fact in FACTS:
                command += ["--fact", fact]
            environment = dict(os.environ, MMJB_DECIDER=decider, MMJB_DECIDER_MODEL=args.llm_model)
            done = subprocess.run(command, capture_output=True, text=True, env=environment)
            try:
                results.append(json.loads(done.stdout))
            except ValueError:
                print(f"{decider} run {repeat + 1} gave no JSON: {done.stderr[-300:]}", file=sys.stderr)
        if results:
            rows.append((decider, results))
    server.shutdown()
    print("| decision model | verified | median seconds | median cost | median steps | runs that used the fallback |")
    print("| --- | --- | --- | --- | --- | --- |")
    for decider, results in rows:
        verified = sum(1 for r in results if r["outcome"] == "verified")
        seconds = statistics.median(r["seconds"] for r in results)
        cost = statistics.median(r["cost"] for r in results)
        steps = statistics.median(len(r["steps"]) for r in results)
        fallbacks = sum(1 for r in results if r["fallback_calls"] > 0)
        print(f"| {decider} | {verified} of {len(results)} | {seconds:.1f} | ${cost:.4f} | {steps:.0f} | {fallbacks} of {len(results)} |")


if __name__ == "__main__":
    main()
