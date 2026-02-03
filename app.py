import streamlit as st

st.set_page_config(page_title="AI SCM RL System", page_icon="📦", layout="centered")

st.title("📦 AI Supply Chain Decision System")
st.markdown("### Reinforcement Learning Based Smart Decision Engine")

# -------- USER SELECTS STATE --------
st.subheader("Select Current Perspective")

state_map = {
    "Inventory Shortage": "s1",
    "Logistics Delay": "s2",
    "Warehouse Congestion": "s3",
    "Demand Spike": "s4"
}

perspective = st.selectbox("Choose situation:", list(state_map.keys()))
current_state = state_map[perspective]

st.info(f"📊 Current Situation: {perspective}")

# -------- RL CORE --------
gamma = 0.9

actions = {
    's1': ['a1', 'a2'],
    's2': ['a3', 'a4'],
    's3': ['a5', 'a6'],
    's4': ['a7', 'a8']
}

action_meanings = {
    'a1': "Order from PRIMARY supplier",
    'a2': "Order from BACKUP supplier",
    'a3': "Switch courier service",
    'a4': "Wait for delivery",
    'a5': "Reroute trucks",
    'a6': "Hold shipments",
    'a7': "Increase production",
    'a8': "Import goods"
}

rewards = {
    ('s1','a1'):5, ('s1','a2'):2,
    ('s2','a3'):4, ('s2','a4'):1,
    ('s3','a5'):5, ('s3','a6'):2,
    ('s4','a7'):5, ('s4','a8'):3
}

policy = {
    ('s1','a1'):0.6, ('s1','a2'):0.4,
    ('s2','a3'):0.7, ('s2','a4'):0.3,
    ('s3','a5'):0.65,('s3','a6'):0.35,
    ('s4','a7'):0.6, ('s4','a8'):0.4
}

# -------- COMPUTE Qπ --------
q_pi = {}
for s in actions:
    for a in actions[s]:
        q_pi[(s,a)] = rewards[(s,a)] + gamma * 0

# -------- AI DECISION --------
if st.button("🔍 Get AI Decision"):

    best_action = max(actions[current_state], key=lambda a: q_pi[(current_state,a)])
    best_reward = q_pi[(current_state,best_action)]

    st.success("### 🤖 AI Recommended Action")
    st.write(action_meanings[best_action])

    st.write("### 📊 Action-Value Function Qπ")
    for a in actions[current_state]:
        st.write(f"{action_meanings[a]} : {q_pi[(current_state,a)]}")

    v_pi = sum(policy[(current_state,a)] * q_pi[(current_state,a)] for a in actions[current_state])

    st.write("### 📈 State-Value Function Vπ")
    st.write(f"Expected Return: {round(v_pi,2)}")

    st.write("### 🧠 Policy (π)")
    for a in actions[current_state]:
        st.write(f"P({a}|{current_state}) = {policy[(current_state,a)]}")

st.markdown("---")
st.caption("AI SCM Decision System using Reinforcement Learning")
