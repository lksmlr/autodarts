import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import io


from triangulation.triangulate import DartScorer
from triangulation.detect import detect


scorer = DartScorer()


def _get_score():
    points = detect()
    result = scorer.calculate_throw(points)
    multiplier = result["multiplier"]
    base_score = result["base_score"]

    score = 0
    if multiplier == "single":
        score = base_score
    elif multiplier == "double":
        score = 2 * base_score
    elif multiplier == "triple":
        score = 3 * base_score
    elif multiplier == "single bull":
        score = base_score
    elif multiplier == "double bull":
        score = 2 * base_score

    return {
        "position": result["position"],
        "score": score,
        "multiplier": multiplier,
        "base_score": base_score,
    }


def create_dartboard_visualization(hit_position=None, score_info=None):
    dartboard_geometry = {
        "outer_double": 170.0,
        "inner_double": 162.0,
        "outer_triple": 107.0,
        "inner_triple": 99.0,
        "outer_bull": 15.9,
        "inner_bull": 6.35,
    }
    segments = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    # Draw rings
    ring_radii = list(dartboard_geometry.values())
    for radius in ring_radii:
        circle = plt.Circle((0, 0), radius, fill=False, color="white", linewidth=1.5)
        ax.add_patch(circle)

    # Draw segment lines
    rotation_offset = np.pi / 2 - np.pi / 20
    for i in range(20):
        angle = i * (2 * np.pi / 20) + rotation_offset
        r1_start, r1_end = (
            dartboard_geometry["outer_bull"],
            dartboard_geometry["inner_triple"],
        )
        ax.plot(
            [r1_start * np.cos(angle), r1_end * np.cos(angle)],
            [r1_start * np.sin(angle), r1_end * np.sin(angle)],
            "w-",
            alpha=0.8,
            linewidth=1.5,
        )
        r2_start, r2_end = (
            dartboard_geometry["outer_triple"],
            dartboard_geometry["inner_double"],
        )
        ax.plot(
            [r2_start * np.cos(angle), r2_end * np.cos(angle)],
            [r2_start * np.sin(angle), r2_end * np.sin(angle)],
            "w-",
            alpha=0.8,
            linewidth=1.5,
        )

    # Draw segment numbers
    text_radius = dartboard_geometry["outer_double"] + 20
    for i, segment_value in enumerate(segments):
        angle = np.pi / 2 - i * (2 * np.pi / 20)
        ax.text(
            text_radius * np.cos(angle),
            text_radius * np.sin(angle),
            str(segment_value),
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="white",
        )

    if hit_position:
        x, y = hit_position
        y = y
        x = x
        ax.scatter(
            x,
            y,
            s=150,
            color="yellow",
            marker="o",
            zorder=10,
            edgecolor="white",
            linewidth=2,
        )

    if score_info:
        score_text = (
            f"Score: {score_info['score']}\n"
            f"Segment: {score_info['base_score']}\n"
            f"Multiplier: {score_info['multiplier'].title()}"
        )
        ax.text(
            0.5,
            -0.05,
            score_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            color="white",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#404040", alpha=0.9),
        )

    ax.set_xlim(-220, 220)
    ax.set_ylim(-220, 220)
    ax.set_aspect("equal")
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", bbox_inches="tight", pad_inches=0.1, transparent=True
    )
    plt.close(fig)
    buf.seek(0)
    return buf


# --- STREAMLIT APP ---
def reset_game():
    st.session_state.reset_trigger = True


st.set_page_config(page_title="AutoDart", layout="wide")
st.title("🎯 AutoDart – Multiplayer")

if st.session_state.get("reset_trigger"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

if "players" not in st.session_state:
    st.markdown("## 🧑‍🤝‍🧑 Select Number of Players")
    num_players = st.number_input(
        "How many players?", min_value=1, max_value=6, value=2
    )
    if st.button("Start Game"):
        st.session_state.players = [
            {
                "name": f"Player {i + 1}",
                "score": 501,
                "history": [],
                "round": 1,
                "finished": False,
            }
            for i in range(num_players)
        ]
        st.session_state.current_player_index = 0
        st.rerun()
    else:
        st.stop()

players = st.session_state.players
current_idx = st.session_state.current_player_index
current_player = players[current_idx]

st.markdown("## 📊 Player Overview")
cols = st.columns(len(players))
for idx, (col, player) in enumerate(zip(cols, players)):
    if idx == current_idx:
        col.markdown(f"### 🟢 **{player['name']} (active)**")
    else:
        col.markdown(f"### ⚪ {player['name']}")
    col.markdown(
        f"<div style='font-size:100px; font-weight:bold; text-align:center;'>{player['score']}</div>",
        unsafe_allow_html=True,
    )
    col.markdown(f"**Round:** {player['round']}")
    if player["finished"]:
        col.success("🏁 Finished!")

st.divider()
st.markdown(
    f"## 🎯 Input for {current_player['name']} – Round {current_player['round']}"
)

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("Take Throw", type="primary"):
        try:
            throw_result = _get_score()
            throw_score = throw_result["score"]
            new_score = current_player["score"] - throw_score

            if new_score < 0 or (
                new_score == 0
                and throw_result["multiplier"] not in ["double", "double bull"]
            ):
                st.warning(
                    "Invalid finish! Score goes below 0 or didn't finish on a double."
                )
                st.session_state.last_throw = {
                    "position": None,
                    "score": 0,
                    "multiplier": "Invalid",
                    "base_score": "Bust",
                }
            else:
                current_player["score"] = new_score
                throw_info = {
                    "Throw": f"{throw_result['base_score']} ({throw_result['multiplier']})",
                    "Points": throw_score,
                }
                if (
                    not current_player["history"]
                    or current_player["history"][-1]["Round"] != current_player["round"]
                ):
                    current_player["history"].append(
                        {
                            "Round": current_player["round"],
                            "Throws": [throw_info],
                            "Round Points": throw_score,
                            "Remaining": current_player["score"],
                        }
                    )
                else:
                    current_player["history"][-1]["Throws"].append(throw_info)
                    current_player["history"][-1]["Round Points"] += throw_score
                    current_player["history"][-1]["Remaining"] = current_player["score"]

                if current_player["score"] == 0:
                    current_player["finished"] = True
                    st.success(f"🏁 {current_player['name']} has finished!")
                    st.balloons()
                st.session_state.last_throw = throw_result
            st.rerun()
        except Exception as e:
            st.error(f"Error during throw detection: {e}")

with col2:
    if "last_throw" in st.session_state:
        last_throw_data = st.session_state.last_throw
        image_buffer = create_dartboard_visualization(
            hit_position=last_throw_data.get("position")
        )
    else:
        image_buffer = create_dartboard_visualization()

    st.image(image_buffer, width=400)

if len(players) == 1 and current_player["finished"]:
    st.success(f"🏁 Game finished! {current_player['name']} has won!")
    st.balloons()
    st.button("🔄 Start New Game", on_click=reset_game)
    st.stop()

st.divider()
st.markdown("## 📋 All Players' History")
for player in players:
    with st.expander(f"{player['name']} – {player['score']} Points"):
        if not player["history"]:
            st.write("No throws yet.")
        for entry in reversed(player["history"]):
            st.markdown(
                f"**Round {entry['Round']} – {entry['Round Points']} Points – Remaining: {entry['Remaining']}**"
            )
            for throw in entry["Throws"]:
                st.write(f"  - {throw['Throw']} → {throw['Points']} Points")

st.divider()
if len(players) > 1:
    if st.button("➡️ Next Player"):
        current_player["round"] += 1
        active_players = [p for p in players if not p["finished"]]
        if len(active_players) <= 1:
            if active_players:
                st.success(
                    f"🏁 Game finished! {active_players[0]['name']} is the winner!"
                )
            else:
                st.success("🎉 All players have finished!")
            st.balloons()
            st.button("🔄 Start New Game", on_click=reset_game)
            st.stop()
        else:
            if "last_throw" in st.session_state:
                del st.session_state.last_throw
            next_idx = current_idx
            while True:
                next_idx = (next_idx + 1) % len(players)
                if not players[next_idx]["finished"]:
                    st.session_state.current_player_index = next_idx
                    break
            st.rerun()
