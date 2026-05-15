#app.py
"""
NutriLens — Streamlit Frontend (app.py)
Run with:  streamlit run app.py
Requires the FastAPI backend (main.py) running at http://localhost:8000
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# ── Config ─────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="NutriLens 🍽️",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ─────────────────────────────────────────────────
for key, default in {
    "logged_in": False,
    "username": "",
    "page": "analyze",
    "last_analysis": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Helpers ────────────────────────────────────────────────────────────────

DISEASE_OPTIONS = ["None", "Diabetes", "Hypertension", "Obesity"]

STATUS_COLOR = {"safe": "🟢", "caution": "🟡", "danger": "🔴"}


def api_post(endpoint: str, **kwargs):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", **kwargs)
        r.raise_for_status()
        return r.json(), None
    except requests.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return None, detail
    except Exception as e:
        return None, str(e)


def api_get(endpoint: str, **kwargs):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", **kwargs)
        r.raise_for_status()
        return r.json(), None
    except requests.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return None, detail
    except Exception as e:
        return None, str(e)


def nutrition_card(label, value, unit=""):
    st.metric(label=label, value=f"{value} {unit}".strip())


# ══════════════════════════════════════════════════════════════════════════
# AUTH PAGES
# ══════════════════════════════════════════════════════════════════════════

def page_login():
    st.title("🍽️ NutriLens — Smart Food Analysis")
    st.subheader("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if not username or not password:
            st.error("Please fill in all fields.")
            return
        data, err = api_post(
            "/api/auth/login",
            json={"username": username, "password": password},
        )
        if err:
            st.error(f"Login failed: {err}")
        else:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(data.get("message", "Logged in!"))
            st.rerun()

    st.markdown("---")
    st.write("Don't have an account?")
    if st.button("Register here"):
        st.session_state.page = "register"
        st.rerun()


def page_register():
    st.title("🍽️ NutriLens — Create Account")
    with st.form("register_form"):
        username = st.text_input("Choose a username")
        password = st.text_input("Choose a password", type="password")
        password2 = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Register")

    if submitted:
        if not username or not password:
            st.error("Please fill in all fields.")
            return
        if password != password2:
            st.error("Passwords do not match.")
            return
        data, err = api_post(
            "/api/auth/register",
            json={"username": username, "password": password},
        )
        if err:
            st.error(f"Registration failed: {err}")
        else:
            st.success(data.get("message", "Registered! Please log in."))
            st.session_state.page = "login"
            st.rerun()

    if st.button("← Back to Login"):
        st.session_state.page = "login"
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# ANALYZE PAGE
# ══════════════════════════════════════════════════════════════════════════

def page_analyze():
    st.title("🔍 Analyze Food")
    st.write(f"Logged in as **{st.session_state.username}**")

    # ── Upload & Settings ──────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        uploaded_file = st.file_uploader(
            "Upload a food photo",
            type=["jpg", "jpeg", "png", "webp"],
            help="Snap or upload a photo of your meal.",
        )
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded image", use_column_width=True)

    with col_right:
        disease = st.selectbox(
            "Do you have a medical condition?",
            DISEASE_OPTIONS,
            help="We'll tailor dietary advice to your condition.",
        )
        analyze_btn = st.button("🔬 Analyze", type="primary", use_container_width=True)

    if not uploaded_file:
        st.info("Please upload a food image to begin analysis.")
        return

    # ── Run Analysis ───────────────────────────────────────────────────────
    if analyze_btn:
        with st.spinner("Analysing your meal…"):
            file_bytes = uploaded_file.getvalue()
            files = {"file": (uploaded_file.name, BytesIO(file_bytes), uploaded_file.type)}
            data_form = {"disease": disease, "username": st.session_state.username}
            result, err = api_post("/api/analyze", files=files, data=data_form)

        if err:
            st.error(f"Analysis failed: {err}")
            return

        st.session_state.last_analysis = result
        st.session_state.last_disease = disease

    # ── Display Results ────────────────────────────────────────────────────
    result = st.session_state.get("last_analysis")
    disease = st.session_state.get("last_disease", "None")

    if not result:
        return

    per_food = result.get("per_food_data", [])
    if not per_food:
        st.warning("No food items detected.")
        return

    st.markdown("---")
    st.subheader(f"Detected {result.get('count', len(per_food))} food item(s)")

    # Tab per food item
    tab_labels = [f"{fd['emoji']} {fd['label'].title()}" for fd in per_food]
    tabs = st.tabs(tab_labels) if len(per_food) > 1 else [st.container()]

    for tab, fd in zip(tabs, per_food):
        with tab:
            _render_food_item(fd, disease)

    # ── Log a Meal ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📝 Log a Meal")
    if per_food:
        food_options = [f["label"].title() for f in per_food]
        selected_idx = st.selectbox("Select food to log", range(len(food_options)),
                                    format_func=lambda i: food_options[i])
        fd_to_log = per_food[selected_idx]
        nut = fd_to_log["nutrition"]

        if st.button("✅ Log this meal", type="primary"):
            payload = {
                "username":  st.session_state.username,
                "food_name": fd_to_log["label"],
                "caption":   fd_to_log["caption"],
                "calories":  nut["calories"],
                "protein":   nut["protein"],
                "carbs":     nut["carbs"],
                "fat":       nut["fat"],
                "portion":   fd_to_log["detected_portion"],
            }
            log_result, err = api_post("/api/meals/log", json=payload)
            if err:
                st.error(f"Logging failed: {err}")
            else:
                st.success("Meal logged successfully! 🎉")


def _render_food_item(fd: dict, disease: str):
    """Render nutrition card + AI recommendation for a single food."""
    status_icon = STATUS_COLOR.get(fd.get("status", "safe"), "🟢")
    st.markdown(f"### {fd['emoji']} {fd['label'].title()}  {status_icon}")
    st.caption(f"Confidence: {fd['confidence']:.0%} | Caption: {fd['caption']}")

    # Tags
    tags = fd.get("tags", [])
    if tags:
        tag_html = " ".join(
            f"<span style='background:#{'4caf50' if t['color']=='green' else 'ff9800' if t['color']=='yellow' else 'f44336'};color:white;padding:2px 8px;border-radius:12px;font-size:0.8em;margin-right:4px'>{t['label']}</span>"
            for t in tags
        )
        st.markdown(tag_html, unsafe_allow_html=True)

    st.markdown("")

    # Portion slider with live rescaling
    base_portion  = fd.get("base_portion", 100)
    init_portion  = fd.get("detected_portion", base_portion)
    base_nut      = fd.get("base_nutrition", fd.get("nutrition", {}))

    portion = st.slider(
        "Adjust portion (g)",
        min_value=25, max_value=500,
        value=int(init_portion), step=25,
        key=f"portion_{fd['index']}",
    )

    # Fetch scaled nutrition from API
    scaled_nut, err = api_get(
        "/api/scale",
        params={
            "base_calories":    base_nut.get("calories", 0),
            "base_protein":     base_nut.get("protein", 0),
            "base_carbs":       base_nut.get("carbs", 0),
            "base_fat":         base_nut.get("fat", 0),
            "base_portion":     base_nut.get("portion", base_portion),
            "current_portion":  portion,
            "disease":          disease,
            "food_name":        fd["label"],
        },
    )
    if err or not scaled_nut:
        scaled_nut = fd["nutrition"]  # fallback

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("🔥 Calories",  f"{scaled_nut['calories']} kcal")
    with c2: st.metric("💪 Protein",   f"{scaled_nut['protein']} g")
    with c3: st.metric("🌾 Carbs",     f"{scaled_nut['carbs']} g")
    with c4: st.metric("🧈 Fat",       f"{scaled_nut['fat']} g")

    # Macro donut chart
    macro_fig = go.Figure(go.Pie(
        labels=["Protein", "Carbs", "Fat"],
        values=[scaled_nut["protein"], scaled_nut["carbs"], scaled_nut["fat"]],
        hole=0.55,
        marker_colors=["#4caf50", "#2196f3", "#ff9800"],
        textinfo="label+percent",
    ))
    macro_fig.update_layout(
        showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=220,
    )
    st.plotly_chart(macro_fig, use_container_width=True, key=f"pie_{fd['index']}")

    # AI Recommendation
    with st.expander("🤖 AI Dietary Recommendation", expanded=True):
        st.write(fd.get("ai_rec", "No recommendation available."))


# ══════════════════════════════════════════════════════════════════════════
# HISTORY PAGE
# ══════════════════════════════════════════════════════════════════════════

def page_history():
    st.title("📋 Meal History")
    username = st.session_state.username

    data, err = api_get(f"/api/meals/history/{username}")
    if err:
        st.error(f"Could not load history: {err}")
        return
    if not data:
        st.info("No meals logged yet. Go analyse some food! 🍴")
        return

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp", ascending=False)

    st.dataframe(
        df[["emoji", "food_name", "calories", "protein", "carbs", "fat", "portion", "timestamp"]].rename(
            columns={"emoji": "", "food_name": "Food", "calories": "Calories (kcal)",
                     "protein": "Protein (g)", "carbs": "Carbs (g)", "fat": "Fat (g)",
                     "portion": "Portion (g)", "timestamp": "Time"}
        ),
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# ANALYTICS PAGE
# ══════════════════════════════════════════════════════════════════════════

def page_analytics():
    st.title("📊 Nutrition Analytics")
    username = st.session_state.username
    disease  = st.selectbox("Filter by condition", DISEASE_OPTIONS, key="analytics_disease")

    data, err = api_get(f"/api/meals/analytics/{username}", params={"disease": disease})
    if err:
        st.error(f"Could not load analytics: {err}")
        return
    if not data or data.get("empty"):
        st.info("No data yet — log some meals first!")
        return

    stats = data["stats"]

    # ── KPI row ────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("🍽️ Total Meals",   stats["total_meals"])
    k2.metric("🔥 Avg Calories",  f"{stats['avg_calories']} kcal")
    k3.metric("💪 Avg Protein",   f"{stats['avg_protein']} g")
    k4.metric("🌾 Avg Carbs",     f"{stats['avg_carbs']} g")
    k5.metric("🧈 Avg Fat",       f"{stats['avg_fat']} g")
    k6.metric("❤️ Health Score",  f"{stats['health_score']} / 100")

    st.markdown("---")

    # ── Charts ─────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🗓️ Calorie Trend (last 7 meals)")
        calorie_df = pd.DataFrame(data["calorie_trend"])
        if not calorie_df.empty:
            fig = px.bar(calorie_df, x="day", y="calories",
                         color_discrete_sequence=["#4caf50"],
                         labels={"day": "Day", "calories": "Calories (kcal)"})
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🥧 Macro Distribution (all time)")
        macros = data["macros"]
        fig2 = go.Figure(go.Pie(
            labels=["Carbs", "Protein", "Fat"],
            values=[macros["carbs"], macros["protein"], macros["fat"]],
            hole=0.5,
            marker_colors=["#2196f3", "#4caf50", "#ff9800"],
        ))
        fig2.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Daily trend line ───────────────────────────────────────────────────
    st.subheader("📈 Daily Nutrition Trend")
    daily_df = pd.DataFrame(data["daily_trend"])
    if not daily_df.empty:
        fig3 = px.line(daily_df, x="date", y=["calories", "protein", "carbs", "fat"],
                       labels={"value": "Amount", "variable": "Nutrient", "date": "Date"},
                       color_discrete_map={"calories": "#f44336", "protein": "#4caf50",
                                           "carbs": "#2196f3", "fat": "#ff9800"})
        fig3.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Top foods & recent meals ───────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🏆 Top Foods")
        top_df = pd.DataFrame(data["top_foods"])
        if not top_df.empty:
            fig4 = px.bar(top_df, x="count", y="food", orientation="h",
                          color_discrete_sequence=["#9c27b0"],
                          labels={"count": "Times logged", "food": "Food"})
            fig4.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig4, use_container_width=True)

    with col4:
        st.subheader("🕐 Recent Meals")
        recent = data.get("recent_meals", [])
        for meal in recent:
            icon  = STATUS_COLOR.get(meal.get("status", "safe"), "🟢")
            emoji = meal.get("emoji", "🍽️")
            name  = meal.get("food_name", "Unknown").title()
            cal   = meal.get("calories", 0)
            ts    = meal.get("timestamp", "")[:16].replace("T", " ")
            st.markdown(
                f"{icon} {emoji} **{name}** — {cal} kcal  <small style='color:grey'>{ts}</small>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════
# WEEKLY INSIGHT PAGE
# ══════════════════════════════════════════════════════════════════════════

def page_weekly_insight():
    st.title("🗓️ Weekly Insight")
    st.write("Get a personalised weekly diet recommendation based on your logged meals.")

    if st.button("✨ Generate Weekly Insight", type="primary"):
        with st.spinner("Generating insight…"):
            result, err = api_post(
                "/api/meals/weekly-insight",
                json={"username": st.session_state.username},
            )
        if err:
            st.error(f"Could not generate insight: {err}")
        else:
            st.success("Here's your personalised weekly insight:")
            st.write(result.get("insight", "No insight returned."))


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR NAV
# ══════════════════════════════════════════════════════════════════════════

def sidebar_nav():
    with st.sidebar:
        st.title("🍽️ NutriLens")
        st.markdown(f"**{st.session_state.username}**")
        st.markdown("---")
        pages = {
            "🔍 Analyze":        "analyze",
            "📋 History":        "history",
            "📊 Analytics":      "analytics",
            "🗓️ Weekly Insight": "weekly",
        }
        for label, key in pages.items():
            if st.button(label, use_container_width=True):
                st.session_state.page = key
                st.rerun()
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username  = ""
            st.session_state.page      = "login"
            st.session_state.last_analysis = None
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════

def main():
    if not st.session_state.logged_in:
        if st.session_state.page == "register":
            page_register()
        else:
            page_login()
        return

    sidebar_nav()

    page = st.session_state.page
    if page == "analyze":
        page_analyze()
    elif page == "history":
        page_history()
    elif page == "analytics":
        page_analytics()
    elif page == "weekly":
        page_weekly_insight()
    else:
        page_analyze()


if __name__ == "__main__":
    main()