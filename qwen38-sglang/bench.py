import json, time, urllib.request
URL = "http://127.0.0.1:8888/v1/chat/completions"
PROMPT = "请以科学的方式介绍太阳系中八大行星的特点，分点列出。"
def run(label, extra):
    body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 400, "temperature": 0.7}
    body.update(extra)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        d = json.loads(resp.read())
    dt = time.time() - t0
    u = d.get("usage", {})
    comp = u.get("completion_tokens", 0); rea = u.get("reasoning_tokens", 0)
    print(label + ": wall=" + str(round(dt,1)) + "s tokens=" + str(comp) + " (reasoning=" + str(rea) + ") -> " + str(round(comp/dt,1)) + " tok/s (avg incl TTFT)")

run("NON-THINKING 400tok", {"chat_template_kwargs": {"enable_thinking": False}})
run("THINKING      400tok", {})
run("NON-THINKING 400tok (repeat)", {"chat_template_kwargs": {"enable_thinking": False}})
