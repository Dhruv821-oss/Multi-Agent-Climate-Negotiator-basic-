import sys
import os

# ---------------- PATH FIX ----------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ---------------- IMPORTS ----------------
from flask import Flask, render_template, jsonify, request
import numpy as np
import importlib.util
news_feed = []
# ---------------- SAFE JSON CONVERTER ----------------
def to_python(obj):
    import numpy as np

    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.astype(float).tolist()

    return obj

def generate_news(env, rewards):
    news = []

    avg_reduction = np.mean(list(env.last_reductions.values()))
    inequality = np.var(list(env.last_reductions.values()))

    # 🌍 Cooperation
    if avg_reduction > 0.7:
        news.append("🤝 Strong global cooperation emerging")

    # ⚠️ Low cooperation
    if avg_reduction < 0.3:
        news.append("⚠️ Nations failing to cooperate")

    # 🔥 Climate danger
    if env.global_temp > 3:
        news.append("🔥 Global temperature entering danger zone")

    # 💸 Budget collapse
    if env.carbon_budget < 20:
        news.append("💸 Carbon budget critically low")

    # 🤝 Coalitions
    for coalition in env.coalitions:
        if len(coalition) > 1:
            news.append(f"🌍 Coalition formed: {', '.join(coalition)}")

    # ⚖️ Inequality
    if inequality > 0.2:
        news.append("⚖️ Rising inequality among nations")

    return news
# ---------------- LOAD ENV ----------------
env_path = os.path.join(BASE_DIR, "env", "climate_env.py")

spec = importlib.util.spec_from_file_location("climate_env", env_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

ClimateParallelEnv = module.ClimateParallelEnv

# ---------------- APP INIT ----------------
app = Flask(__name__)

# ---------------- GLOBAL STATE ----------------
current_config = {
    "n_agents": 3,
    "agent_names": None,
    "agent_types": None,
    "shocks": {}
}

env, obs = None, None
env_B, obs_B = None, None
step_count = 0

# ---------------- METRICS ----------------
history = {
    "temperature": [],
    "avg_reduction": [],
    "inequality": []
}


# ---------------- ENV CREATION ----------------
def create_env(config=None):
    if config is None:
        config = current_config

    env = ClimateParallelEnv(config)
    obs, _ = env.reset()
    return env, obs


# initialize
env, obs = create_env()
env_B, obs_B = create_env()


# ---------------- RANDOM POLICY ----------------
def random_actions(target_env):
    return {
        agent: np.random.rand(3).astype(np.float32)
        for agent in target_env.agents
    }


# ---------------- SCENARIOS ----------------
SCENARIOS = {
    "normal": {},
    "climate_crisis": {"shocks": {"temperature_spike": 1.5}},
    "economic_collapse": {"shocks": {"finance_crash": True}},
    "random_disaster": {"shocks": {"random_disaster": True}}
}


# ---------------- EQUILIBRIUM ----------------
def detect_equilibrium():
    if len(history["temperature"]) < 10:
        return False

    recent = history["temperature"][-10:]
    return (max(recent) - min(recent)) < 0.01


# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/sim")
def sim():
    return render_template("index.html")

@app.route("/chat")
def chat_ui():
    return render_template("chat.html")

from google import genai

client = genai.Client(api_key="AIzaSyB2A0rGt268wPQjJHGCOzM4r6lzvZ2kfL0")


@app.route("/chat_api", methods=["POST"])
def chat_api():
    data = request.json
    user_msg = data.get("message", "")

    # -------- CONTEXT --------
    context = {
        "temperature": float(getattr(env, "global_temp", 0)),
        "carbon_budget": float(getattr(env, "carbon_budget", 0)),
        "reductions": getattr(env, "last_reductions", {}),
        "finance": getattr(env, "last_finance", {}),
        "coalitions": getattr(env, "coalitions", []),
        "reward_breakdown": getattr(env, "last_reward_breakdown", {})
    }

    prompt = f"""
You are an expert AI analyzing a multi-agent climate negotiation system.

Context:
{context}

User Question:
{user_msg}

Explain clearly using reasoning, not generic statements.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",   # ✅ correct in new SDK
        contents=prompt
    )

    return jsonify({"reply": response.text})

# -------- VIDEO PAGE --------
@app.route("/video")
def video_page():
    return render_template("video.html")


# -------- PROMPT (from current sim state) --------
@app.route("/video_prompt")
def video_prompt():
    temp = getattr(env, "global_temp", 1.0)
    coalitions = getattr(env, "coalitions", [])
    reductions = getattr(env, "last_reductions", {})

    if reductions:
        avg_reduction = sum(reductions.values()) / len(reductions)
    else:
        avg_reduction = 0.5

    if avg_reduction > 0.7:
        tone = "cooperative agreement, hopeful"
    elif avg_reduction < 0.3:
        tone = "tense, conflict, disagreement"
    else:
        tone = "mixed negotiation, cautious cooperation"

    prompt = f"""
Cinematic scene of a global climate summit.

Atmosphere: {tone}
Temperature crisis level: {round(temp, 2)}
Coalitions: {coalitions}

Show:
- world leaders at a round table
- emotional negotiation dynamics
- subtle gestures showing trust or tension
- dramatic lighting, realistic style, 4K, slow camera movement
"""

    return jsonify({"prompt": prompt})


# -------- GENERATE VIDEO (Veo) --------

@app.route("/generate_video", methods=["POST"])
def generate_video():
    data = request.json
    prompt = data.get("prompt", "")

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=prompt
        )

        # ⚠️ Depending on API, adjust extraction:
        video_url = getattr(resp, "url", None) or str(resp)

        return jsonify({"video_url": video_url})

    except Exception as e:
        return jsonify({"error": str(e)})
# ---------------- CONFIGURE ----------------
@app.route("/configure", methods=["POST"])
def configure():
    global env, obs, env_B, obs_B, step_count, current_config

    data = request.json or {}

    # ---------------- PARSE AGENT NAMES ----------------
    names_raw = data.get("agent_names")

    if names_raw:
        names = [n.strip() for n in names_raw.split(",") if n.strip()]
    else:
        names = None

    # ---------------- DETERMINE NUMBER OF AGENTS ----------------
    if names:
        n_agents = len(names)
    else:
        n_agents = int(data.get("n_agents", 3))
        names = [f"agent_{i}" for i in range(n_agents)]

    # ---------------- PARSE AGENT TYPES ----------------
    types_raw = data.get("agent_types")

    if types_raw:
        types_list = [t.strip().lower() for t in types_raw.split(",") if t.strip()]

        # 🔥 Normalize values
        def normalize_type(t):
            if t in ["developed", "dev", "rich"]:
                return "developed"
            return "developing"

        # 🔥 Fix length mismatch
        if len(types_list) < n_agents:
            types_list += ["developing"] * (n_agents - len(types_list))

        types_list = types_list[:n_agents]

        agent_types = {
            names[i]: normalize_type(types_list[i])
            for i in range(n_agents)
        }
    else:
        agent_types = None  # env will auto-assign

    # ---------------- SCENARIO ----------------
    scenario = data.get("scenario", "normal")
    scenario_config = SCENARIOS.get(scenario, {})

    # ---------------- SHOCKS ----------------
    user_shocks = data.get("shocks", {})
    combined_shocks = {
        **scenario_config.get("shocks", {}),
        **user_shocks
    }

    # ---------------- FINAL CONFIG ----------------
    current_config = {
        "n_agents": n_agents,
        "agent_names": names,
        "agent_types": agent_types,
        "shocks": combined_shocks
    }

    # ---------------- RECREATE ENV ----------------
    env, obs = create_env(current_config)
    env_B, obs_B = create_env(current_config)

    step_count = 0

    # reset metrics
    for k in history:
        history[k].clear()

    # ---------------- DEBUG (KEEP THIS FOR NOW) ----------------
    print("\n=== CONFIG APPLIED ===")
    print("Agents:", env.agents)
    print("Types:", env.agent_types)
    print("Shocks:", combined_shocks)
    print("======================\n")

    # ---------------- RESPONSE ----------------
    return jsonify({
        "status": "configured",
        "agents": env.agents,
        "agent_types": env.agent_types,
        "scenario": scenario,
        "shocks": combined_shocks
    })

# ---------------- STEP ----------------
@app.route("/step")
def step():
    global env, obs, step_count, news_feed
    
    try:
        actions = random_actions(env)

        obs, rewards, _, _, _ = env.step(actions)
        step_count += 1

        # ---------- METRICS ----------
        avg_reduction = np.mean(list(env.last_reductions.values()))
        inequality = np.var(list(env.last_reductions.values()))

        history["temperature"].append(env.global_temp)
        history["avg_reduction"].append(avg_reduction)
        history["inequality"].append(inequality)

        # ---------- NEWS ----------
        new_news = generate_news(env, rewards)
        news_feed.extend(new_news)

        # keep last 10
        news_feed[:] = news_feed[-10:]

        # ---------- RESPONSE ----------
        data = {
            "step": step_count,
            "temperature": env.global_temp,
            "carbon_budget": env.carbon_budget,

            "agents": env.agents,
            "rewards": rewards,

            "coalitions": env.coalitions,
            "final_reductions": env.last_reductions,
            "finance": env.last_finance,

            "memory": env.memory,
            "trust": env.trust,

            "equilibrium": detect_equilibrium(),

            # 🔥 IMPORTANT: include news
            "news": news_feed
            
        }

        return jsonify(to_python(data))

    except Exception as e:
        return jsonify({"error": str(e)})

# ---------------- METRICS ----------------
@app.route("/metrics")
def metrics():
    return jsonify(to_python(history))


# ---------------- COMPARISON ----------------
@app.route("/compare")
def compare():
    global env, env_B, obs, obs_B

    try:
        actions_A = random_actions(env)
        actions_B = random_actions(env_B)

        obs, _, _, _, _ = env.step(actions_A)
        obs_B, _, _, _, _ = env_B.step(actions_B)

        data = {
            "world_A": {
                "temperature": env.global_temp,
                "coalitions": env.coalitions
            },
            "world_B": {
                "temperature": env_B.global_temp,
                "coalitions": env_B.coalitions
            }
        }

        return jsonify(to_python(data))

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/explain")
def explain():
    return jsonify(to_python({
        "rewards": env.last_reward_breakdown,
        "reductions": env.last_reductions,
        "coalitions": env.coalitions,
        "temperature": env.global_temp,
        "budget": env.carbon_budget
    }))
@app.route("/explain_ui")
def explain_ui():
    return render_template("explain.html")
# ---------------- RESET ----------------
@app.route("/reset")
def reset():
    global env, obs, env_B, obs_B, step_count

    env, obs = create_env(current_config)
    env_B, obs_B = create_env(current_config)
    step_count = 0

    for k in history:
        history[k].clear()

    return jsonify({"status": "reset"})


# ---------------- AUTO RUN ----------------
@app.route("/run_auto")
def run_auto():
    global env, obs, step_count

    try:
        for _ in range(5):
            actions = random_actions(env)
            obs, _, _, _, _ = env.step(actions)
            step_count += 1

        return jsonify({
            "step": step_count,
            "temperature": float(env.global_temp)
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# ---------------- HEALTH ----------------
@app.route("/health")
def health():
    return jsonify({"status": "running"})


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)