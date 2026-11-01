import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import swarm_sim as s

st.set_page_config(page_title="Swarm Drones for Disaster Relief", layout="wide")
st.title("Swarm Drones for Disaster Relief")
st.caption("Compare a random walk, an A* sector sweep and a pheromone swarm "
           "searching for survivors on a 50x50 disaster map.")

NAME = {"random": "Random walk", "astar": "A* sector sweep", "swarm": "Pheromone swarm"}
COL = {"random": "#9a9a9a", "astar": "#1f77b4", "swarm": "#d95f02"}

with st.sidebar:
    st.header("Scenario")
    kind = st.radio("Disaster type", ["earthquake", "flood"], format_func=str.capitalize)
    seed = st.number_input("Map seed", 0, 999, 0)
    n_drones = st.slider("Drones", 1, 10, 5)
    steps = st.slider("Time steps", 50, 300, 300, step=10)
    strategies = st.multiselect("Strategies", list(NAME), default=list(NAME),
                                format_func=NAME.get)

@st.cache_data(show_spinner="Running simulation...")
def simulate(kind, seed, n_drones, steps, strat):
    scn = s.make_scenario(kind, seed)
    r = s.run(scn, strat, n_drones=n_drones, T=steps, seed=seed, record=True)
    return scn, r

if not strategies:
    st.info("Pick at least one strategy in the sidebar.")
    st.stop()

runs = {k: simulate(kind, seed, n_drones, steps, k) for k in strategies}
scn = runs[strategies[0]][0]

st.subheader("Summary")
cols = st.columns(len(strategies))
for c, k in zip(cols, strategies):
    r = runs[k][1]
    c.markdown(f"**{NAME[k]}**")
    c.metric("Survivors found", f"{r.n_found}/{r.n_surv}")
    c.metric("Coverage", f"{r.coverage*100:.0f}%")
    c.metric("Energy per survivor found", f"{r.energy_per_found:.0f}")

st.subheader("Map snapshot")
max_t = min(len(runs[k][1].positions) for k in strategies) - 1
t = st.slider("Time step", 0, max_t, min(120, max_t))
fig, axes = plt.subplots(1, len(strategies), figsize=(4.2 * len(strategies), 4), squeeze=False)
for ax, k in zip(axes[0], strategies):
    r = runs[k][1]
    img = np.where(np.isfinite(scn.cost), np.where(scn.cost > 1, 0.8, 1.0), 0.25)
    rgb = np.dstack([img, img, img]).astype(float)
    cov = r.cov_frames[t]
    rgb[cov & np.isfinite(scn.cost)] *= np.array([0.78, 0.9, 1.0])
    ax.imshow(rgb, interpolation="nearest")
    fnd, sv = r.found_frames[t], scn.survivors
    ax.scatter(sv[~fnd, 1], sv[~fnd, 0], c="red", s=16, marker="x", label="undetected")
    ax.scatter(sv[fnd, 1], sv[fnd, 0], c="green", s=18, label="found")
    ax.scatter(*scn.base[::-1], c="k", marker="s", s=35, label="base")
    ax.scatter(r.positions[t][:, 1], r.positions[t][:, 0], c="orange",
               edgecolors="k", s=32, marker="^", label="drones")
    ax.set_title(f"{NAME[k]}: {int(fnd.sum())}/{r.n_surv} found", fontsize=10)
    ax.axis("off")
axes[0][0].legend(fontsize=7, loc="lower left")
plt.tight_layout()
st.pyplot(fig)

st.subheader("Survivors found over time")
fig2, ax2 = plt.subplots(figsize=(8, 3.2))
for k in strategies:
    r = runs[k][1]
    ax2.plot(r.found_curve / r.n_surv * 100, color=COL[k], label=NAME[k], lw=2)
ax2.axvline(t, color="k", ls=":", lw=1)
ax2.set_xlabel("Time step"); ax2.set_ylabel("Survivors found (%)")
ax2.grid(alpha=.3); ax2.legend(frameon=False)
st.pyplot(fig2)
