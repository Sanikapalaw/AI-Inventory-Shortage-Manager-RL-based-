import streamlit as st

st.set_page_config(page_title="AI SCM RL System", page_icon="📦", layout="centered")

st.title("📦 AI Supply Chain Decision System")
st.markdown("### Reinforcement Learning Based Smart Decision Engine")

# -------- USER PERSPECTIVE DROPDOWN --------
st.subheader("Select Current Perspective")

perspective = st.selectbox(
    "Choose a situation:",
    ["Inventory Shortage", "Logistics Delay", "Warehouse Congestion", "Demand Spike"]
)

st.info(f"📊 Current Situation: {perspective}")

# -------- RL CORE (same logic for all) --------
gamma = 0.9
states = ['s1', 's2']

actions = {
    's1': ['a1', 'a2'],
    's2': []
}

# Rewards for all scenarios
rewards = {
    ('s1','a1'): 5,
    ('s1','a2'): 2
}

policy = {
    ('s1','a1'): 0.6,
    ('s1','a2'): 0.4
}

# Action value
q_pi = {}
for action in actions['s1']:
    q_pi[('s1', action)] = rewards[('s1', action)] + gamma * 0

# State value
v_pi = {}
v_pi['s1'] = (
    policy[('s1','a1')] * q_pi[('s1','a1')] +
    policy[('s1','a2')] * q_pi[('s1','a2')]
)
v_pi['s2'] = 0

# -------- AI DECISION BUTTON --------
if st.button("🔍 Get AI Decision"):

    decision_map = {
        "Inventory Shortage": ("Order from PRIMARY supplier", "Order from BACKUP supplier"),
        "Logistics Delay": ("Switch courier service", "Wait for delivery"),
        "Warehouse Congestion": ("Reroute trucks", "Hold shipments"),
        "Demand Spike": ("Increase production", "Import goods")
    }

    best_action = "a1" if q_pi[('s1','a1')] > q_pi[('s1','a2')] else "a2"

    decision_text = decision_map[perspective][0] if best_action == "a1" else decision_map[perspective][1]

    st.success("### 🤖 AI Recommended Action")
    st.write(decision_text)

    st.write("### 📊 Action-Value Function Qπ")
    st.write(f"a1: {q_pi[('s1','a1')]}")
    st.write(f"a2: {q_pi[('s1','a2')]}")

    st.write("### 📈 State-Value Function Vπ")
    st.write(f"Expected Return: {v_pi['s1']}")

    st.write("### 🧠 Policy")
    st.write("P(a1|s1) = 0.6")
    st.write("P(a2|s1) = 0.4")

st.markdown("---")
st.caption("AI SCM Decision System using Reinforcement Learning")
