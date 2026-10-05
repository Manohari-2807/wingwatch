"""WingWatch: student pilot landing-performance prediction (Streamlit).

Run locally:  pip install -r requirements.txt  then  streamlit run app.py
Passwords live in st.secrets (never in this file). See README.md.
"""
import plotly.graph_objects as go
import streamlit as st

import core

st.set_page_config(page_title="WingWatch", page_icon="🛩️", layout="wide")

# Light grey-blue background, navy text, amber warnings
st.markdown("""
<style>
.stApp { background:#dfe5ea; color:#14213d; }
h1,h2,h3 { color:#14213d; }
section[data-testid="stSidebar"] { background:#cfd8df; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training the model (first visit only)...")
def load_model():
    return core.train()


def show_chart(fig):
    """Works on both old and new Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def get_passwords():
    try:
        return dict(st.secrets["passwords"])
    except Exception:
        return None


# ------------------------------------------------------------------- login
def login_screen():
    st.title("WingWatch")
    st.write("Log in to see landing performance predictions.")
    passwords = get_passwords()
    if not passwords:
        st.error("No passwords are set up. Add a [passwords] section to the app's Secrets "
                 "(see README.md), then reload.")
        return
    with st.form("login"):
        role_label = st.radio("Login type", ["Student login", "Faculty login"], horizontal=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
    if submitted:
        role = "faculty" if role_label == "Faculty login" else "student"
        user = core.check_login(role, username, password, passwords)
        if user:
            st.session_state["user"] = user
            st.rerun()
        else:
            st.error("Wrong username, password, or login type.")


# --------------------------------------------------------------- dashboard
def dashboard(user):
    forest, medians, table = load_model()

    # Access rule: students only ever get their own record; faculty get all.
    records = core.visible_students(user)
    if not records:
        st.error("No student record is linked to this account.")
        return

    with st.sidebar:
        st.write(f"Signed in as **{user['username']}** ({user['role']})")
        if st.button("Log out"):
            del st.session_state["user"]
            st.rerun()
        if user["role"] == "faculty":
            key = st.selectbox("Student", list(records), format_func=lambda k: records[k]["name"])
        else:
            key = next(iter(records))
    rec = records[key]
    history = rec["history"]

    st.title("WingWatch")
    st.caption("Predicted landing performance for student pilots")
    tab_dash, tab_model = st.tabs(["Dashboard", "Model"])

    with tab_dash:
        top = st.container()
        bottom = st.container()   # filled first so slider values feed the prediction above

        inputs = {k: rec[k] for k, *_ in core.FACTORS}
        inputs.update({k: rec[k] for k, *_ in core.HOURS})

        # Faculty: this section is never rendered, so faculty cannot change inputs.
        if core.can_adjust_inputs(user):
            with bottom:
                st.subheader("Adjust inputs")
                st.caption("Try what-if changes. Your saved record isn't changed.")
                cols = st.columns(4)
                i = 0
                for k, label, _focus in core.FACTORS:
                    with cols[i % 4]:
                        inputs[k] = st.slider(label, 0.0, 1.0, float(rec[k]), 0.01, key=f"{key}_{k}")
                    i += 1
                for k, label, lo, hi, step in core.HOURS:
                    with cols[i % 4]:
                        if isinstance(step, float):
                            inputs[k] = st.slider(label, float(lo), float(hi), float(rec[k]), float(step), key=f"{key}_{k}")
                        else:
                            inputs[k] = st.slider(label, int(lo), int(hi), int(rec[k]), int(step), key=f"{key}_{k}")
                    i += 1

        pred = core.predict(forest, medians, inputs, history)

        with top:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.subheader("Student profile")
                st.write(f"**Student:** {rec['name']}")
                st.write(f"**Flight hours:** {inputs['flight_hours']}")
                st.write(f"**Simulator hours:** {inputs['sim_hours']}")
                st.write(f"**Landings:** {inputs['total_landings']}")
                st.write(f"**IFR landings:** {inputs['ifr_landings']}")
                st.write(f"**Last landing score:** {history[-1]}")
            with c2:
                st.subheader("Predicted landing score")
                color = "#2f7d5b" if pred["score"] >= 70 else "#1f5f8b" if pred["score"] >= 55 else "#c98a00"
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=pred["score"],
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": color}},
                ))
                fig.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=10),
                                  paper_bgcolor="rgba(0,0,0,0)")
                show_chart(fig)
                st.write(f"**{core.category(pred['score'])}**")
                st.caption(f"Likely range {pred['low']} to {pred['high']}")
            with c3:
                st.subheader("Key factors")
                for k, label, _focus in core.FACTORS:
                    v = float(inputs[k])
                    st.progress(min(max(v, 0.0), 1.0),
                                text=f"{label}: {v:.2f}" + ("  (below threshold)" if v < core.THRESHOLD else ""))

            d1, d2 = st.columns([5, 7])
            with d1:
                st.subheader("Score trend")
                xs = list(range(1, len(history) + 1))
                tf = go.Figure()
                tf.add_trace(go.Scatter(x=xs, y=history, mode="lines+markers", name="Past landings",
                                        line=dict(color="#1f5f8b")))
                tf.add_trace(go.Scatter(x=[xs[-1], xs[-1] + 1], y=[history[-1], pred["score"]],
                                        mode="lines+markers", name="Predicted next",
                                        line=dict(color="#14213d", dash="dash")))
                tf.update_layout(height=260, margin=dict(l=20, r=20, t=10, b=30),
                                 yaxis=dict(range=[0, 100], title="Score"),
                                 xaxis=dict(title="Landing number"), paper_bgcolor="rgba(0,0,0,0)")
                show_chart(tf)
            with d2:
                st.subheader("Training focus")
                weak = core.weak_factors(inputs)
                if weak:
                    names = ", ".join(f for _, _, f in weak)
                    detail = " ".join(f"{label} is {inputs[k]:.2f}, below the {core.THRESHOLD:.2f} threshold."
                                      for k, label, _ in weak)
                    st.warning(f"**Recommend more practice: {names}**\n\n{detail}")
                else:
                    st.success("No focus areas flagged. All key factors are at or above "
                               f"{core.THRESHOLD:.2f}.")
                if user["role"] == "faculty":
                    st.caption("Read-only view. Faculty accounts can't change student inputs.")

    with tab_model:
        st.subheader("Model comparison")
        st.write("Held-out 20% of the training data. The Random Forest is the prediction engine.")
        st.dataframe(table, hide_index=True)
        st.caption("Trained on synthetic data for this prototype, so these scores are not "
                   "evidence about real landings.")


# -------------------------------------------------------------------- main
user = st.session_state.get("user")
if user:
    dashboard(user)
else:
    login_screen()
