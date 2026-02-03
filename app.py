import streamlit as st

st.set_page_config(page_title="AI Supply Chain RL System", page_icon="📦", layout="centered")

st.title("📦 AI Inventory Shortage Manager")
st.markdown("### Reinforcement Learning based Decision System")

# ---------------- RL CORE ----------------

gamma = 0.9

states = ['s1', 's2']   # s1 = Stock Low, s2 = Restored

actions = {
    's1': ['a1', 'a2'],
    's2': []
}

rewards = {
    ('s1','a1'): 5,   # Primary supplier
    ('s1','a2'): 2    # Backup supplier
}

policy = {
    ('s1','a1'): 0.6,
    ('s1','a2'): 0.4
}

# Action-value function qπ(s,a)
q_pi = {}
for action in actions['s1']:
    q_pi[('s1', action)] = rewards[('s1', action)] + gamma * 0

# State-value function vπ(s)
v_pi = {}
v_pi['s1'] = (
    policy[('s1','a1')] * q_pi[('s1','a1')] +
    policy[('s1','a2')] * q_pi[('s1','a2')]
)
v_pi['s2'] = 0

# ---------------- STREAMLIT UI ----------------

st.subheader("Current Situation")
st.info("⚠️ Inventory is critically LOW")

if st.button("🔍 Check AI Decision"):
    if q_pi[('s1','a1')] > q_pi[('s1','a2')]:
        decision = "Order from PRIMARY supplier"
    else:
        decision = "Order from BACKUP supplier"

    st.success("### 🤖 AI Decision")
    st.write(f"**Recommended Action:** {decision}")

    st.write("### 📊 Action-Value Function Qπ")
    st.write(f"Primary Supplier (a1): {q_pi[('s1','a1')]}")
    st.write(f"Backup Supplier (a2): {q_pi[('s1','a2')]}")

    st.write("### 📈 State-Value Function Vπ")
    st.write(f"Expected Return from current state: {v_pi['s1']}")

    st.write("### 🧠 Policy")
    st.write("P(a1|s1) = 0.6")
    st.write("P(a2|s1) = 0.4")

st.markdown("---")
st.caption("Reinforcement Learning Demo | SCM AI System")
