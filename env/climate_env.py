import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import Box


class ClimateParallelEnv(ParallelEnv):
    metadata = {"name": "climate_marl_v9"}

    def __init__(self, config=None):
        if config is None:
            config = {}

        # ---------- AGENTS ----------
        self.n_agents = config.get("n_agents", 3)

        agent_names = config.get("agent_names")
        if agent_names is None:
            self.agents = [f"agent_{i}" for i in range(self.n_agents)]
        else:
            self.agents = agent_names
            self.n_agents = len(self.agents)

        self.agent_index = {a: i for i, a in enumerate(self.agents)}

        # ---------- TYPES ----------
        agent_types = config.get("agent_types")

        if agent_types is None:
            self.agent_types = {
                a: ("developed" if i < self.n_agents // 2 else "developing")
                for i, a in enumerate(self.agents)
            }
        else:
            self.agent_types = agent_types

        # ---------- EMISSIONS ----------
        self.base_emissions = np.array([
            12.0 if self.agent_types[a] == "developed" else 8.0
            for a in self.agents
        ], dtype=np.float32)

        # ---------- SHOCKS ----------
        self.shock_config = config.get("shocks", {})

        # ---------- REWARD ----------
        self.alpha = 2.0
        self.beta = 3.0
        self.gamma = 1.5
        self.delta = 2.0
        self.lambda_f = 1.0
        self.theta = 1.0

        self.max_rounds = 3

        # ---------- SPACES ----------
        self._obs_spaces = {
            a: Box(low=0.0, high=100.0, shape=(5,), dtype=np.float32)
            for a in self.agents
        }

        self._act_spaces = {
            a: Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
            for a in self.agents
        }

    # ---------------- SPACES ----------------
    def observation_space(self, agent):
        return self._obs_spaces[agent]

    def action_space(self, agent):
        return self._act_spaces[agent]

    # ---------------- RESET ----------------
    def reset(self, seed=None, options=None):
        self.global_temp = 1.0
        self.carbon_budget = 100.0
        self.emissions = self.base_emissions.copy()

        self.trust = np.ones((self.n_agents, self.n_agents), dtype=np.float32)
        self.memory = np.zeros(self.n_agents, dtype=np.float32)

        self.coalitions = []
        self.last_reductions = {}
        self.last_finance = {}
        self.last_reward_breakdown = {}

        obs = {a: self._get_obs(a) for a in self.agents}
        return obs, {}

    # ---------------- SHOCKS ----------------
    def apply_shocks(self):
        if "temperature_spike" in self.shock_config:
            self.global_temp += self.shock_config["temperature_spike"]

        if self.shock_config.get("random_disaster", False):
            if np.random.rand() < 0.1:
                self.global_temp += 0.5

    # ---------------- STEP ----------------
    def step(self, actions):
        self.last_reward_breakdown = {}

        actions = {k: np.clip(v, 0, 1) for k, v in actions.items()}

        reductions = {k: float(v[0]) for k, v in actions.items()}
        finance = {k: float(v[1]) for k, v in actions.items()}
        willingness = {k: float(v[2]) for k, v in actions.items()}

        # ---------- NEGOTIATION ----------
        current_r = reductions.copy()
        current_f = finance.copy()

        for _ in range(self.max_rounds):
            avg_r = np.mean(list(current_r.values()))
            next_r, next_f = {}, {}

            for a in self.agents:
                i = self.agent_index[a]

                accept = sum(
                    self.trust[i][self.agent_index[o]] * willingness[a] > 0.5
                    for o in self.agents if o != a
                )

                if accept >= (self.n_agents - 1) / 2:
                    next_r[a] = current_r[a]
                    next_f[a] = current_f[a]
                else:
                    next_r[a] = (current_r[a] + avg_r) / 2
                    next_f[a] = current_f[a] * (0.8 + 0.2 * willingness[a])

            current_r, current_f = next_r, next_f

        final_r = current_r
        final_f = current_f

        self.last_reductions = final_r
        self.last_finance = final_f

        # ---------- MEMORY ----------
        for a in self.agents:
            i = self.agent_index[a]
            self.memory[i] = 0.8 * self.memory[i] + 0.2 * final_r[a]

        # ---------- COALITIONS ----------
        coalitions = []
        assigned = set()

        for a in self.agents:
            if a in assigned:
                continue

            group = [a]
            assigned.add(a)

            for b in self.agents:
                if b in assigned or a == b:
                    continue

                i = self.agent_index[a]
                j = self.agent_index[b]

                if abs(final_r[a] - final_r[b]) < 0.2 and self.trust[i][j] > 1.0:
                    group.append(b)
                    assigned.add(b)

            coalitions.append(group)

        self.coalitions = coalitions

        coalition_bonus = {
            a: 0.2 * len(group)
            for group in coalitions for a in group
        }

        # ---------- SHOCKS ----------
        self.apply_shocks()

        # ---------- CLIMATE ----------
        emissions = [
            (1 - final_r[a]) * self.emissions[self.agent_index[a]]
            for a in self.agents
        ]

        total_emissions = sum(emissions)
        self.global_temp += 0.01 * total_emissions
        self.carbon_budget -= total_emissions

        finance_bonus = sum(final_f.values()) / self.n_agents
        avg_action = np.mean(list(final_r.values()))
        inequality = np.var(list(final_r.values()))

        overshoot = abs(self.carbon_budget) * 0.05 if self.carbon_budget < 0 else 0

        # ---------- TRUST ----------
        for a in self.agents:
            for b in self.agents:
                if a == b:
                    continue

                i = self.agent_index[a]
                j = self.agent_index[b]

                diff = abs(final_r[a] - final_r[b])
                self.trust[i][j] = np.clip(
                    self.trust[i][j] + 0.02*(1-diff) - 0.01*diff,
                    0, 2
                )

        # ---------- REWARDS ----------
        rewards = {}
        for a in self.agents:
            rewards[a] = self._compute_reward(
                a, final_r[a], avg_action, finance_bonus,
                inequality, overshoot, coalition_bonus.get(a, 0)
            )

        obs = {a: self._get_obs(a) for a in self.agents}

        return obs, rewards, {a: False for a in self.agents}, {a: False for a in self.agents}, {a: {} for a in self.agents}

    # ---------------- REWARD ----------------
    def _compute_reward(self, agent, action, avg, finance, inequality, penalty, coalition):
        i = self.agent_index[agent]
        t = self.agent_types[agent]

        econ = (1 - action) * (0.8 if t == "developed" else 1.2)
        finance_term = finance * (1.2 if t == "developed" else 0.8)

        breakdown = {
            "economic": self.alpha * econ,
            "climate": -self.beta * (self.global_temp ** 2),
            "cooperation": self.gamma * avg,
            "free_riding": -self.delta * max(0, avg - action),
            "finance": self.lambda_f * finance_term,
            "inequality": -self.theta * inequality,
            "trust": 0.5 * np.mean(self.trust[i]),
            "memory": 0.5 * self.memory[i],
            "coalition": coalition,
            "penalty": -penalty
        }

        self.last_reward_breakdown[agent] = breakdown

        reward = sum(breakdown.values())
        return reward

    # ---------------- OBS ----------------
    def _get_obs(self, agent):
        i = self.agent_index[agent]

        return np.array([
            self.global_temp,
            self.carbon_budget,
            self.emissions[i],
            np.mean(self.trust[i]),
            self.memory[i]
        ], dtype=np.float32)