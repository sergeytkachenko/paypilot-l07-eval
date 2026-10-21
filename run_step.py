import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
STAND = os.environ.get("SERVICE_URL", "http://localhost:8000")
MOCK = os.environ.get("MOCK_URL", "http://localhost:1080")
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
TYPES = {"boolean": bool, "object": dict, "string": str, "number": (int, float), "array": list}


class RunError(RuntimeError):
    pass


def _call(base, method, path, body=None, headers=None, timeout=300):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with _opener.open(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise RunError(f"{method} {base}{path} -> {e.code}: {e.read().decode('utf-8')[:300]}") from e
    except urllib.error.URLError as e:
        raise RunError(f"cannot reach {base} ({e.reason})") from e


def _settings_header(settings):
    return {"X-Stand-Settings": json.dumps(settings)} if settings else {}


def doctor(scenarios):
    health = _call(STAND, "GET", "/health")
    known = set(_call(STAND, "GET", "/api/_test/defects").get("all_defects", []))
    needed = {d for s in scenarios for c in s["conditions"].values()
              for d in (c.get("defects") or "").split(",") if d}
    missing = sorted(needed - known)
    if missing:
        raise RunError(f"the stand does not know {missing}: update paypilot-stand to the commit named in README")
    if health.get("provider") in (None, "mock"):
        raise RunError("the stand runs LLM_PROVIDER=mock: the live step needs a real model behind the proxy")
    _call(MOCK, "PUT", "/mockserver/retrieve?type=ACTIVE_EXPECTATIONS", {})
    return health


def _mock_requests():
    return _call(MOCK, "PUT", "/mockserver/retrieve?type=REQUESTS", {"path": ".*/chat/completions"})


def _clear_mock_log():
    _call(MOCK, "PUT", "/mockserver/clear?type=LOG", {"path": ".*"})


def _tool_spans(tree):
    return [c for c in tree.get("children", []) if c["name"].startswith("tool.")]


def _llm_spans(tree):
    return [c for c in tree.get("children", []) if c["name"] == "llm.call"]


def _subset(actual, expected):
    return all(actual.get(k) == v for k, v in expected.items())


def check(assertion, tree):
    kind = assertion["kind"]
    tools = _tool_spans(tree)
    if kind in ("tool_called", "tool_not_called", "tool_call_count"):
        hits = [t for t in tools if t["attributes"]["tool.name"] == assertion["tool"]
                and _subset(t["attributes"].get("tool.arguments") or {}, assertion.get("args", {}))]
        if kind == "tool_called":
            seen = [t["attributes"].get("tool.arguments") for t in tools
                    if t["attributes"]["tool.name"] == assertion["tool"]]
            return bool(hits), f"{assertion['tool']} calls: {seen or 'none'}"
        if kind == "tool_not_called":
            return not hits, f"{assertion['tool']}{assertion.get('args', {})} called {len(hits)}x"
        return len(hits) == assertion["expected"], f"{assertion['tool']} called {len(hits)}x, expected {assertion['expected']}"
    if kind == "tool_order":
        names = [t["attributes"]["tool.name"] for t in tools]
        first, second = assertion["before"]
        if second not in names:
            return True, f"{second} not called, order holds: {names}"
        if first not in names:
            return False, f"{second} called without {first}: {names}"
        ok = names.index(first) < names.index(second)
        return ok, f"{first} {'before' if ok else 'after'} {second}: {names}"
    if kind == "tool_sequence":
        names = [t["attributes"]["tool.name"] for t in tools]
        problems = [f"{n} not called" for n in assertion.get("must_include", []) if n not in names]
        problems += [f"{n} should not be called" for n in assertion.get("must_exclude", []) if n in names]
        return not problems, "; ".join(problems) or f"tools {names}"
    if kind == "max_tool_calls":
        names = [t["attributes"]["tool.name"] for t in tools]
        return len(names) <= assertion["max"], f"{len(names)} tool calls {names}, max {assertion['max']}"
    if kind == "tool_result_schema":
        results = [t["attributes"].get("tool.result") for t in tools
                   if t["attributes"]["tool.name"] == assertion["tool"]]
        if not results:
            return False, f"{assertion['tool']} was not called"
        bad = []
        for result in results:
            for key, type_name in assertion["required"].items():
                if not isinstance(result, dict) or not isinstance(result.get(key), TYPES[type_name]):
                    bad.append(f"{key}: expected {type_name}, got {json.dumps(result)[:120]}")
        return not bad, "; ".join(bad) or "result matches the contract"
    if kind == "state_row":
        rows = _call(STAND, "GET", f"/api/_test/state/{assertion['table']}")["rows"]
        matched = [r for r in rows if _subset(r, assertion.get("where", {}))]
        problems = []
        if "expected_count" in assertion and len(matched) != assertion["expected_count"]:
            problems.append(f"{len(matched)} rows, expected {assertion['expected_count']}")
        for row in matched:
            for key, value in assertion.get("expect_fields", {}).items():
                if row.get(key) != value:
                    problems.append(f"{key}={row.get(key)!r}, expected {value!r}")
        shown = [{k: r.get(k) for k in ("transaction_id", "account_id", "amount", "currency")} for r in matched]
        return not problems, "; ".join(problems) or f"rows {shown}"
    if kind == "cache_read_min":
        calls = _llm_spans(tree)[assertion["from_call"]:]
        if not calls:
            return False, f"the step made {len(_llm_spans(tree))} model call(s): nothing to read from cache"
        reads = [c["attributes"].get("gen_ai.usage.cache_read_input_tokens", 0) for c in calls]
        return all(r >= assertion["min"] for r in reads), f"cache_read per call from #{assertion['from_call'] + 1}: {reads}"
    raise RunError(f"unknown assertion kind {kind!r}")


def run_once(scenario, settings):
    _call(STAND, "POST", "/api/_test/reset", {})
    _clear_mock_log()
    headers = _settings_header(settings)
    session = None
    scripted_calls = 0
    for step, turn in enumerate(scenario["turns"], 1):
        res = _call(STAND, "POST", "/chat", {"message": turn["user"], "session_id": session,
                                             "customer_id": scenario["customer_id"]}, headers)
        session = res["session_id"]
        if turn.get("live"):
            tree = _call(STAND, "GET", f"/api/_test/traces/{res['request_id']}")
            return step, res, tree, scripted_calls
        scripted = turn["model"][-1]["text"]
        if res["answer"] != scripted:
            raise RunError(f"step {step} was meant to come from the mock, but the answer differs: "
                           f"{res['answer'][:120]!r}. Is the stand behind the proxy (HTTPS_PROXY)?")
        scripted_calls += len(turn["model"])
    raise RunError("scenario has no live step")


def run(scenario, condition, settings, runs):
    rows = []
    for n in range(1, runs + 1):
        step, res, tree, scripted_calls = run_once(scenario, settings)
        verdicts = [(a, *check(a, tree)) for a in scenario["assertions"]]
        llm = _llm_spans(tree)
        rows.append({
            "run": n, "live_step": step, "passed": all(v[1] for v in verdicts),
            "verdicts": [{"kind": a["kind"], "ok": ok, "detail": d} for a, ok, d in verdicts],
            "answer": res["answer"],
            "tools": [t["attributes"]["tool.name"] for t in _tool_spans(tree)],
            "model_calls": len(llm),
            "cache_read": [c["attributes"].get("gen_ai.usage.cache_read_input_tokens", 0) for c in llm],
            "cost_usd": res["usage"].get("cost_usd"),
            "mock_answered": scripted_calls,
            "mock_seen": len(_mock_requests()),
        })
        print(_line(scenario, condition, rows[-1]))
    return rows


def _line(scenario, condition, row):
    mark = "PASS" if row["passed"] else "FAIL"
    head = (f"{scenario['id']} {condition:<8} run {row['run']} {mark}  step {row['live_step']}: "
            f"{row['model_calls']} model call(s), tools {row['tools']}, "
            f"cache_read {row['cache_read']}, cost ${row['cost_usd']}; "
            f"mock answered {row['mock_answered']} of {row['mock_seen']} model requests")
    details = [f"    {'ok ' if v['ok'] else 'BAD'} {v['kind']}: {v['detail']}" for v in row["verdicts"]]
    return "\n".join([head, *details])


def main():
    parser = argparse.ArgumentParser(description="Test one step of a multi-turn chat: mocked history, live last step.")
    parser.add_argument("--scenario", help="S1..S5; all scenarios when omitted")
    parser.add_argument("--condition", default="all", help="baseline | defect | all")
    parser.add_argument("--runs", type=int, help="runs per condition; default 1, or the scenario's runs with --gate")
    parser.add_argument("--gate", help="daily | release: only scenarios with this gate")
    parser.add_argument("--json", help="write raw results to this file")
    args = parser.parse_args()

    scenarios = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "scenarios").glob("*.json"))]
    if args.scenario:
        scenarios = [s for s in scenarios if s["id"].lower() == args.scenario.lower()]
        if not scenarios:
            raise SystemExit(f"no scenario {args.scenario}")
    if args.gate:
        scenarios = [s for s in scenarios if s.get("gate") == args.gate]
        if not scenarios:
            raise SystemExit(f"no scenarios with gate {args.gate}")
    try:
        health = doctor(scenarios)
        print(f"stand {STAND}: provider {health.get('provider')}, mock {MOCK}")
        results = []
        for scenario in scenarios:
            print(f"\n== {scenario['id']} · {scenario['title']}")
            for condition, settings in scenario["conditions"].items():
                if args.condition not in ("all", condition):
                    continue
                runs = args.runs or (scenario.get("runs", 1) if args.gate else 1)
                rows = run(scenario, condition, settings, runs)
                print(f"   {condition}: {sum(r['passed'] for r in rows)}/{len(rows)} passed")
                results.append({"scenario": scenario["id"], "condition": condition,
                                "settings": settings, "runs": rows})
    except RunError as e:
        raise SystemExit(f"ERROR: {e}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
