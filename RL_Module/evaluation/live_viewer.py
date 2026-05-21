"""
Live viewer: reads step CSV log every 0.5 seconds and renders the latest
student state in a pygame window.

Run in a SEPARATE terminal while training runs in another:

  Terminal 1: python -m RL_Module.main_experiment --algo PPO --seed 42
  Terminal 2: python -m RL_Module.evaluation.live_viewer --algo PPO --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
import pygame

from RL_Module import config

COLORS = {
    "bg": (15, 15, 26),
    "panel": (22, 22, 46),
    "confused": (254, 188, 46),
    "frustrated": (255, 95, 87),
    "bored": (136, 136, 187),
    "engaged": (40, 200, 128),
    "bar_k": (58, 123, 213),
    "bar_e": (17, 153, 142),
    "bar_f": (255, 75, 43),
    "bar_c": (254, 188, 46),
    "bar_b": (136, 136, 187),
    "active_chip": (42, 31, 90),
    "text": (180, 180, 220),
    "dim": (80, 80, 140),
    "purple": (124, 106, 247),
}

ACTION_NAMES = [
    "easier_q", "harder_q", "hint", "motivation", "scaffold",
    "pacing", "reflection", "strategy", "autonomy", "explanation",
]

EMOTION_NAMES = ["confused", "bored", "frustrated", "engaged"]


def get_latest_row(algo: str, seed: int = 42):
    path = config.LOGS_DIR / f"steps_{algo}_seed{seed}.csv"
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if len(df) == 0:
            return None
        return df.iloc[-1]
    except Exception:
        return None


def draw_bar(surf, font, name, value, color, x, y, w=280, h=16):
    pygame.draw.rect(surf, (30, 30, 60), (x, y, w, h), border_radius=4)
    fill_w = max(0, min(w, int(w * float(value))))
    pygame.draw.rect(surf, color, (x, y, fill_w, h), border_radius=4)
    label = font.render(f"{name}  {float(value):.2f}", True, COLORS["text"])
    surf.blit(label, (x + w + 8, y))


def run_viewer(algo: str, seed: int) -> None:
    pygame.init()
    win = pygame.display.set_mode((660, 520))
    pygame.display.set_caption(f"Live Viewer - {algo} seed={seed}")
    clock = pygame.time.Clock()
    font_sm = pygame.font.SysFont("monospace", 11)
    font_md = pygame.font.SysFont("monospace", 14)
    font_lg = pygame.font.SysFont("monospace", 20, bold=True)

    reward_history: list = []

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        row = get_latest_row(algo, seed)
        win.fill(COLORS["bg"])

        if row is None:
            msg = font_md.render(
                f"Waiting for {config.LOGS_DIR}/steps_{algo}_seed{seed}.csv ...",
                True,
                COLORS["dim"],
            )
            win.blit(msg, (20, 240))
            pygame.display.flip()
            clock.tick(1)
            continue

        step = int(row.get("step", 0))
        episode = int(row.get("episode", 0))
        action_id = int(row.get("action_id", 0))
        reward = float(row.get("reward", 0))
        cum_reward = float(row.get("cumulative_reward", 0))
        knowledge = float(row.get("knowledge", 0))
        engagement = float(row.get("engagement", 0))
        frustration = float(row.get("frustration", 0))
        confusion = float(row.get("confusion", 0))
        boredom = float(row.get("boredom", 0))
        emotion_id = int(float(row.get("emotion_id", 3)))
        emotion = EMOTION_NAMES[min(emotion_id, 3)]
        p_flag = bool(row.get("persistent_flag", False))
        reason = str(row.get("explainer_reason", ""))[:80]

        reward_history.append(cum_reward)
        if len(reward_history) > 60:
            reward_history.pop(0)

        pygame.draw.rect(win, COLORS["panel"], (0, 0, 660, 32))
        header = font_md.render(
            f"{algo}  |  Episode {episode}  |  Step {step}  "
            f"|  {'PERSISTENT' if p_flag else ''}",
            True,
            COLORS["dim"],
        )
        win.blit(header, (10, 8))

        pygame.draw.rect(win, COLORS["panel"], (10, 40, 140, 90), border_radius=8)
        emo_color = COLORS.get(emotion, COLORS["text"])
        emo_label = font_lg.render(emotion.upper(), True, emo_color)
        win.blit(emo_label, (18, 58))
        win.blit(font_sm.render("FER OUTPUT", True, COLORS["dim"]), (18, 44))

        pygame.draw.rect(win, COLORS["panel"], (160, 40, 490, 90), border_radius=8)
        win.blit(font_sm.render("STUDENT STATE", True, COLORS["dim"]), (168, 44))
        bars = [
            ("knowledge", knowledge, COLORS["bar_k"]),
            ("engagement", engagement, COLORS["bar_e"]),
            ("frustration", frustration, COLORS["bar_f"]),
            ("confusion", confusion, COLORS["bar_c"]),
            ("boredom", boredom, COLORS["bar_b"]),
        ]
        for i, (name, val, col) in enumerate(bars):
            draw_bar(win, font_sm, name, val, col, x=168, y=56 + i * 17, w=240)

        pygame.draw.rect(win, COLORS["panel"], (10, 138, 640, 40), border_radius=8)
        win.blit(font_sm.render("ACTIONS", True, COLORS["dim"]), (18, 142))
        for i, name in enumerate(ACTION_NAMES):
            x = 10 + i * 63
            active = i == action_id
            bg = COLORS["active_chip"] if active else (15, 15, 26)
            bd = COLORS["purple"] if active else (50, 50, 90)
            pygame.draw.rect(win, bg, (x, 154, 60, 18), border_radius=4)
            pygame.draw.rect(win, bd, (x, 154, 60, 18), 1, border_radius=4)
            c = (200, 190, 255) if active else (100, 100, 140)
            win.blit(font_sm.render(name[:8], True, c), (x + 2, 157))

        for i, (label, val) in enumerate([
            ("Step reward", f"{reward:.3f}"),
            ("Cumulative", f"{cum_reward:.2f}"),
            ("Episode", str(episode)),
        ]):
            x = 10 + i * 216
            pygame.draw.rect(win, COLORS["panel"], (x, 182, 210, 48), border_radius=8)
            win.blit(font_lg.render(val, True, (200, 190, 255)), (x + 8, 192))
            win.blit(font_sm.render(label, True, COLORS["dim"]), (x + 8, 218))

        pygame.draw.rect(win, COLORS["panel"], (10, 238, 640, 32), border_radius=8)
        win.blit(font_sm.render("WHY: " + reason, True, COLORS["text"]), (18, 248))

        pygame.draw.rect(win, COLORS["panel"], (10, 278, 640, 90), border_radius=8)
        win.blit(
            font_sm.render("CUMULATIVE REWARD (last 60 steps)", True, COLORS["dim"]),
            (18, 284),
        )
        if len(reward_history) > 1:
            mn = min(reward_history)
            mx = max(reward_history)
            rng = max(mx - mn, 0.1)
            pts = [
                (
                    18 + int(i / (len(reward_history) - 1) * 624),
                    358 - int((v - mn) / rng * 60),
                )
                for i, v in enumerate(reward_history)
            ]
            pygame.draw.lines(win, COLORS["purple"], False, pts, 2)

        if "hybrid_agent_used" in row:
            h_agent = str(row["hybrid_agent_used"])
            pygame.draw.rect(win, COLORS["panel"], (10, 376, 640, 28), border_radius=8)
            win.blit(
                font_sm.render(f"Hybrid decision: {h_agent}", True, COLORS["text"]),
                (18, 384),
            )

        pygame.display.flip()
        clock.tick(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="PPO")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_viewer(args.algo, args.seed)
