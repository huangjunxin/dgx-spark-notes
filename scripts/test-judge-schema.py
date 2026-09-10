import subprocess, json, sys, re
VLLM = "http://127.0.0.1:8889/v1/chat/completions"
schema = {
  "type": "json_schema",
  "json_schema": {"name": "CapabilityClassifierDecision", "strict": True,
    "schema": {"type": "object", "additionalProperties": False,
      "required": ["crux","primary_rule","capability_boundary","p_solve"],
      "properties": {
        "crux": {"type":"string","minLength":1},
        "primary_rule": {"type":"string","enum":["SUP-1","SUP-2","SUP-3","SUP-4","SUP-5","UNC-1","UNC-2","LIM-1","LIM-2","none"]},
        "capability_boundary": {"type":"string","enum":["supported","uncertain","unsupported","unmatched"]},
        "p_solve": {"type":"number","minimum":0.0,"maximum":1.0}}}}
SYS = """You are a task-level probability forecaster for a model router.
Forecast SUCCESS = the local 3.5B-active efficient agent completes the whole task correctly on one fresh run.
Procedure: 1) state the crux; 2) pick one primary_rule from SUP-1..SUP-5, UNC-1, UNC-2, LIM-1, LIM-2, none; 3) set capability_boundary to supported/uncertain/unsupported/unmatched; 4) estimate p_solve in [0,1] (probability of SUCCESS, not confidence).
Output EXACTLY one JSON object with only the 4 fields; no commentary."""
TASKS = {
  "easy": "What is 2+2?",
  "medium": "Compute exactly: 17^3 + 23*41 - 999. Show steps.",
  "hard": "Detect a cycle in a linked list with O(1) memory and prove correctness.",
}
def main():
    for name, task in TASKS.items():
        body = json.dumps({"model":"qwen3-8b-awq","messages":[{"role":"system","content":SYS},{"role":"user","content":task}],"response_format":schema,"max_tokens":300,"temperature":0.0,"extra_body":{"chat_template_kwargs":{"enable_thinking":False}}})
        out = subprocess.run(["curl","-s","--max-time","90","-X","POST",VLLM,"-H","content-type: application/json","--data",body],capture_output=True,text=True).stdout
        try:
            d = json.loads(out); ch = d.get("choices",[])
            if not ch:
                print(f"{name:7s} -> error: {(d.get('error',{}) or {}).get('message','?')[:60]}")
                continue
            m = ch[0].get("message",{}); c = (m.get("content") or "").strip()
            try:
                o = json.loads(c)
                ok = all(k in o for k in ("crux","primary_rule","capability_boundary","p_solve"))
                print(f"{name:7s} -> VALID JSON  fields_ok={ok}  p_solve={o.get('p_solve')}  rule={o.get('primary_rule')}  boundary={o.get('capability_boundary')}")
            except Exception:
                print(f"{name:7s} -> NOT-JSON: {c[:100]!r}")
        except Exception as e:
            print(f"{name:7s} -> parsefail raw={out[:80]!r}")
if __name__ == "__main__":
    main()
