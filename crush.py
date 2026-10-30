from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import textwrap
import urllib.error
import urllib.request
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

try:
    import joblib
    from sklearn.tree import DecisionTreeClassifier
    HAS_ML = True
except ImportError:
    print("\n  [Setup] Installing required libraries (scikit-learn, joblib)...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "joblib"])
        import joblib
        from sklearn.tree import DecisionTreeClassifier
        HAS_ML = True
        print("  [Setup] Installed successfully!\n")
    except Exception as _e:
        HAS_ML = False
        joblib = None
        DecisionTreeClassifier = None

SYS_PROMPT = "{sys_prompt}"
ML_DATA = []
raw = ML_DATA
ROUTINE_OBSERVATIONS = []
DRESSCODE_DATA = []
DEFAULT_NIM_MODEL = "nvidia/nemotron-3.5-bolt-e4b"
NIM_TEMPERATURE = 0.4
NIM_MAX_TOKENS = 300




def label(times: int) -> str:
    if times == 0:
        return "no_look"
    elif times == 1:
        return "one_look"
    elif times == 2:
        return "multiple_looks"
    else:
        return "fr_noticed_you"


def train_model(data=None, destination=None):
    if not HAS_ML:
        return None
    if data is None:
        data = raw
    if not data:
        return None
    X = []
    y = []
    for item in data:
        if len(item) == 4:
            m, d, _sit, count = item
            dt = date(2026, m, d)
        elif len(item) == 5:
            yr, m, d, _sit, count = item
            dt = date(yr, m, d)
        else:
            continue
        X.append([dt.weekday()])
        y.append(label(count))
    if not X:
        return None
    clf = DecisionTreeClassifier(random_state=42, max_depth=4)
    clf.fit(X, y)
    if destination and joblib:
        joblib.dump(clf, destination)
    return clf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "crush.joblib"


def get_history_path() -> Path:
    personal = ROOT / "observations_personal.json"
    if personal.exists():
        return personal
    obs = ROOT / "observations.json"
    if not obs.exists():
        try:
            obs.write_text("[]\n", encoding="utf-8")
        except OSError:
            pass
    return obs


def get_dresscode_path() -> Path:
    personal = ROOT / "dresscodes_personal.json"
    if personal.exists():
        return personal
    dress = ROOT / "dresscodes.json"
    if not dress.exists():
        try:
            dress.write_text("[]\n", encoding="utf-8")
        except OSError:
            pass
    return dress


HISTORY_PATH = get_history_path()
DRESSCODE_PATH = get_dresscode_path()
NIM_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_NIM_MODEL = "nvidia/nemotron-3.5-bolt-e4b"
PERSEUS_MESSAGES = []
PREDICTION_COUNTS = {
    "no_look": "0 times",
    "one_look": "1 time",
    "multiple_looks": "2 times",
    "fr_noticed_you": "3 or more times",
}


def clear_screen():
    if os.name == "nt":
        os.system("cls")
    else:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()


class Tone:
    enabled = sys.stdout.isatty()
    pink = "\033[38;5;212m"
    red = "\033[38;5;196m"
    green = "\033[38;5;46m"
    yellow = "\033[38;5;226m"
    violet = "\033[38;5;141m"
    dim = "\033[2m"
    bold = "\033[1m"
    reset = "\033[0m"

    @classmethod
    def paint(cls, value, color=""):
        return f"{color}{value}{cls.reset}" if cls.enabled and color else value


class ExitApp(Exception):
    pass


class ExitChat(Exception):
    pass


def strip_emojis(text: str) -> str:
    emoji_pattern = re.compile(
        r"[\U00010000-\U0010ffff]|[\u2600-\u27bf]|[\u2300-\u23ff]|[\u2b50-\u2b55]|[\ufe00-\ufe0f]|[\u200d]",
        flags=re.UNICODE,
    )
    cleaned = emoji_pattern.sub("", text)
    return re.sub(r"[ \t]+", " ", cleaned).strip()


def read_line(prompt, ctrl_b_exits_chat=False):
    if not ctrl_b_exits_chat or os.name != "nt" or not sys.stdin.isatty():
        try:
            value = input(prompt)
            if ctrl_b_exits_chat and value.strip().lower() in ("\x02", "exit", "quit", "menu", "back"):
                raise ExitChat
            return value
        except EOFError as error:
            raise ExitApp from error
        except KeyboardInterrupt:
            raise

    import msvcrt
    print(prompt, end="", flush=True)
    chars = []
    while True:
        char = msvcrt.getwch()
        if char == "\x03":
            print()
            raise KeyboardInterrupt
        if char in ("\x04", "\x1a"):
            print()
            raise ExitApp
        if char == "\x02":
            if ctrl_b_exits_chat:
                print()
                raise ExitChat
            continue
        if char in ("\r", "\n"):
            print()
            return "".join(chars)
        if char in ("\x00", "\xe0"):
            msvcrt.getwch()
            continue
        if char in ("\b", "\x7f"):
            if chars:
                chars.pop()
                print("\b \b", end="", flush=True)
            continue
        if char.isprintable():
            chars.append(char)
            print(char, end="", flush=True)


def get_nim_key():
    key = os.getenv("NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")
    if key:
        return key.strip()
    env_path = ROOT / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("NIM_API_KEY=") or line.startswith("NVIDIA_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"\'')
                    if val and val != "your_nvidia_api_key_here":
                        return val
        except Exception:
            pass
    if os.name == "nt":
        import winreg
        for subkey in (r"Environment", r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"):
            root = winreg.HKEY_CURRENT_USER if subkey == r"Environment" else winreg.HKEY_LOCAL_MACHINE
            try:
                with winreg.OpenKey(root, subkey) as rk:
                    val, _ = winreg.QueryValueEx(rk, "NIM_API_KEY")
                    if val:
                        return str(val).strip()
            except Exception:
                pass
            try:
                with winreg.OpenKey(root, subkey) as rk:
                    val, _ = winreg.QueryValueEx(rk, "NVIDIA_API_KEY")
                    if val:
                        return str(val).strip()
            except Exception:
                pass
    return None


def parse_date(value: str) -> date | None:
    value = value.strip().lower()
    if not value:
        return None
    today = date.today()
    if value in ("today", "tod"):
        return today
    if value in ("yesterday", "yest"):
        return today - timedelta(days=1)
    if value in ("tomorrow", "tom"):
        return today + timedelta(days=1)

    cleaned = value.replace("/", "-").replace(".", "-").replace(" ", "-")
    cleaned = re.sub(r"-+", "-", cleaned)
    if cleaned.count("-") == 1:
        cleaned = f"{cleaned}-{today.year}"

    for fmt in ("%d-%m-%Y", "%d-%m-%y", "%Y-%m-%d", "%d-%b-%Y", "%d-%B-%Y"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            pass
    return None


def extract_date_and_count(text: str):
    cleaned = text.strip()
    if not cleaned:
        return None, None, None

    target_time = None
    time_match = re.search(r"\b(\d{1,2}:\d{2}\s*(?:am|pm)?)\b", cleaned, re.IGNORECASE)
    if time_match:
        target_time = time_match.group(1).upper()
        cleaned = cleaned[:time_match.start()] + " " + cleaned[time_match.end():]

    tokens = [t.strip("(),;:") for t in re.split(r"[\s,;:]+", cleaned) if t.strip("(),;:")]
    noise = {"looks", "look", "times", "time", "on", "at", "for"}
    tokens = [t for t in tokens if t.lower() not in noise]

    if not tokens:
        return None, None, target_time

    if len(tokens) == 1:
        if tokens[0].isdigit():
            return date.today(), int(tokens[0]), target_time
        d = parse_date(tokens[0])
        if d:
            return d, None, target_time

    target_date = None
    target_count = None
    for t in tokens:
        if target_count is None and t.isdigit():
            target_count = int(t)
            continue
        if target_date is None:
            d = parse_date(t)
            if d:
                target_date = d
                continue

    if target_count is not None and target_date is None:
        target_date = date.today()

    return target_date, target_count, target_time


def extract_date_and_color(text: str) -> tuple[date | None, str | None, str | None]:
    cleaned = text.strip()
    if not cleaned:
        return None, None, None
    if "," in cleaned:
        parts = [p.strip() for p in cleaned.split(",", 1)]
        d1 = parse_date(parts[0])
        d2 = parse_date(parts[1])
        if d1 is not None:
            color = parts[1].strip().lower()
            return d1, (color if color else None), None
        if d2 is not None:
            color = parts[0].strip().lower()
            return d2, (color if color else None), None
        return None, None, f"Invalid date: '{parts[0]}'. Use format DD-MM-YYYY (e.g. 27-10-2026)."
    tokens = cleaned.split()
    if len(tokens) == 1:
        d = parse_date(tokens[0])
        if d is not None:
            return d, None, None
        return date.today(), tokens[0].lower(), None
    d = parse_date(tokens[0])
    if d is not None:
        color = " ".join(tokens[1:]).strip().lower()
        return d, color, None
    d = parse_date(tokens[-1])
    if d is not None:
        color = " ".join(tokens[:-1]).strip().lower()
        return d, color, None
    if any(c.isdigit() for c in tokens[0]) and ("-" in tokens[0] or "/" in tokens[0]):
        return None, None, f"Invalid date: '{tokens[0]}'. Use format DD-MM-YYYY (e.g. 27-10-2026)."
    return date.today(), cleaned.lower(), None


def history():
    if not get_history_path().exists():
        return []
    try:
        data = json.loads(get_history_path().read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def rows():
    added = []
    for item in history():
        try:
            logged = date.fromisoformat(item["date"])
            added.append((logged.year, logged.month, logged.day, 0, int(item["looks"])))
        except (KeyError, TypeError, ValueError):
            pass
    replaced_dates = {(year, month, day) for year, month, day, _, _ in added}
    seed = [record for record in raw if (2026, record[0], record[1]) not in replaced_dates]
    return seed + added


def refresh():
    r = rows()
    if not r:
        return None
    return train_model(r, MODEL_PATH)


def classifier():
    if not HAS_ML:
        return None
    if MODEL_PATH.exists() and joblib:
        try:
            saved = joblib.load(MODEL_PATH)
            if getattr(saved, "n_features_in_", None) == 1:
                return saved
        except Exception:
            pass
    return refresh()


def dresscodes() -> list[dict]:
    if not get_dresscode_path().exists():
        return []
    try:
        data = json.loads(get_dresscode_path().read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def log_dresscode(when: date, color: str) -> bool:
    entries = dresscodes()
    is_overwrite = any(entry.get("date") == when.isoformat() for entry in entries)
    entries = [entry for entry in entries if entry.get("date") != when.isoformat()]
    entries.append({
        "date": when.isoformat(),
        "color": color.strip().lower(),
        "logged_at": datetime.now().isoformat(timespec="seconds"),
    })
    entries.sort(key=lambda item: item["date"])
    get_dresscode_path().write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    return is_overwrite


def predict_color(target_date: date, entries: list[dict]) -> tuple[str | None, str]:
    if not entries:
        return None, "No color history recorded yet."

    target_dow = target_date.weekday()
    dow_colors = []
    all_colors = []

    for item in entries:
        col = (item.get("color") or item.get("outfit", "")).strip().lower()
        if not col:
            continue
        try:
            d = date.fromisoformat(item["date"])
            all_colors.append(col)
            if d.weekday() == target_dow:
                dow_colors.append(col)
        except Exception:
            continue

    if dow_colors:
        counts = Counter(dow_colors)
        top_color, top_cnt = counts.most_common(1)[0]
        pct = int((top_cnt / len(dow_colors)) * 100)
        day_name = target_date.strftime("%A")
        return top_color.capitalize(), f"{pct}% match based on {len(dow_colors)} logged {day_name}(s)"

    if all_colors:
        counts = Counter(all_colors)
        top_color, top_cnt = counts.most_common(1)[0]
        day_name = target_date.strftime("%A")
        return top_color.capitalize(), f"Overall most frequent ({top_cnt}/{len(all_colors)} logs); no {day_name} logs yet"

    return None, "No valid color logs found."


def ask_yn(question: str) -> bool:
    while True:
        resp = read_line(f"  {question} [y/n] > ").strip().lower()
        if resp in ("y", "yes"):
            return True
        if resp in ("n", "no"):
            return False
        print("  Please type 'y' for yes or 'n' for no.\n")


APPROACH_QUESTION_POOL = [
    ("Did she glance back or look in your direction more than once today?", 22),
    ("Was her body orientation angled towards your direction rather than turned away?", 18),
    ("Did you notice her smile or soften her expression when you were nearby?", 20),
    ("Did she hold eye contact for more than a brief accidental split-second?", 22),
    ("Did she look up from her phone or conversation as you walked past?", 18),
    ("Did she seem aware of your presence and receptive rather than closed off?", 18),
    ("Did she look towards you while talking with her friends?", 18),
    ("Is she in an unhurried spot right now (waiting at a stop, lawn, or corridor)?", 20),
    ("Is she alone or with just one approachable friend (not a large crowd)?", 18),
    ("Is she free from intense distractions (not on a call, cramming, or listening to music)?", 20),
    ("Are you in a natural environment where a casual hello feels normal?", 18),
    ("Does she have a few minutes of downtime before her next class or bus?", 18),
    ("Is the setting quiet enough that you won't have to raise your voice or shout?", 16),
    ("Do you have a simple, natural observation or question ready (class, bus, schedule)?", 20),
    ("Have you crossed paths enough times that your face is familiar to her?", 20),
    ("Is there an immediate shared circumstance right now you can casually comment on?", 18),
    ("Can you approach her naturally in her field of view without startling her?", 18),
    ("Is your opener short and low-stakes so she doesn't feel trapped in a long talk?", 18),
    ("Are you feeling calm, composed, and ready to smile naturally?", 16),
    ("Are you totally fine keeping it brief (under 60-90 seconds) with zero pressure?", 18),
    ("Are you ready to exit gracefully with a friendly nod if she seems reserved?", 18),
    ("Are you approaching just to say a friendly hello rather than trying to impress?", 18),
]


def calculate_approach_score(answers: list[bool], weights: list[int], looks_today: int, dress_color_today: str | None) -> int:
    total_w = sum(weights) if weights else 1
    earned_w = sum(w for a, w in zip(answers, weights) if a)
    base = (earned_w / total_w) * 85
    bonus = 0
    if looks_today >= 2:
        bonus += 10
    elif looks_today == 1:
        bonus += 5
    if dress_color_today:
        bonus += 5
    return max(15, min(int(base + bonus), 98))


DOT_HEART = [
    "      .............      ..............     ",
    "    ................    .................   ",
    "   ..................  ...................  ",
    " ..........................................  ",
    "............................................",
    "............................................",
    "............................................",
    " .......................................... ",
    "  ........................................  ",
    "   ......................................   ",
    "     ..................................     ",
    "       ..............................       ",
    "         ..........................         ",
    "           ......................           ",
    "             ..................             ",
    "               ..............               ",
    "                 ..........                 ",
    "                   ......                   ",
    "                     ..                     ",
]


def header():
    columns = shutil.get_terminal_size((80, 24)).columns
    heart_width = len(DOT_HEART[0])
    padding = " " * max(0, (columns - heart_width) // 2)

    print()
    for line in DOT_HEART:
        print(padding + Tone.paint(line, Tone.pink))

    print()
    title = "Crush AI-ML Model - Observation & Prediction CLI ;)"
    title_padding = " " * max(0, (columns - len(title)) // 2)
    print(title_padding + Tone.paint("Crush AI-ML Model", Tone.green + Tone.bold) + Tone.paint(" - Observation & Prediction CLI ", Tone.bold) + Tone.paint(";)", Tone.red))
    print()


def all_observations():
    by_date = {}
    for item in raw:
        if len(item) == 4:
            month, day, _situation, count = item
            yr = 2026
        elif len(item) == 5:
            yr, month, day, _situation, count = item
        d = date(yr, month, day)
        by_date[d] = {"looks": count, "time": None}

    for item in history():
        try:
            d = date.fromisoformat(item["date"])
            t = item.get("time")
            by_date[d] = {"looks": int(item["looks"]), "time": t}
        except (KeyError, TypeError, ValueError):
            continue

    result = []
    for day, data in sorted(by_date.items()):
        record = {
            "date": day.strftime("%d-%m-%Y"),
            "iso_date": day.isoformat(),
            "date_obj": day,
            "in_person_looks": data["looks"],
        }
        if data.get("time"):
            record["time"] = data["time"]
        result.append(record)
    return result


def log_entry(when: date, count: int, time_str: str | None = None):
    entries = history()
    is_overwrite = any(entry.get("date") == when.isoformat() for entry in entries)
    entries = [entry for entry in entries if entry.get("date") != when.isoformat()]
    record = {
        "date": when.isoformat(),
        "looks": int(count),
        "logged_at": datetime.now().isoformat(timespec="seconds"),
    }
    if time_str:
        record["time"] = time_str
    entries.append(record)
    entries.sort(key=lambda item: item["date"])
    get_history_path().write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    refresh()
    return is_overwrite


def predict_for(when: date):
    if not HAS_ML:
        print(f"\n  Forecast for {when:%d-%m-%Y} ({when:%A}):")
        print("  Missing required dependencies (scikit-learn, joblib).")
        print("  Run: pip install -r requirements.txt (or double-click crush.bat)!")
        return
    clf = classifier()
    if clf is None:
        print(f"\n  Forecast for {when:%d-%m-%Y} ({when:%A}):")
        print("  No observation data logged yet. Use [3] Add entry to log your first observation!")
        return
    known = {entry["date_obj"]: entry["in_person_looks"] for entry in all_observations()}
    predicted = clf.predict([[when.weekday()]])[0]
    estimated_text = PREDICTION_COUNTS.get(predicted, str(predicted))
    if predicted in ("multiple_looks", "fr_noticed_you"):
        p_color = Tone.green + Tone.bold
    elif predicted == "one_look":
        p_color = Tone.yellow + Tone.bold
    else:
        p_color = Tone.red + Tone.bold

    print(f"\n  Forecast for {when:%d-%m-%Y} ({when:%A}):")
    print(f"  Estimated eye contact: {Tone.paint(estimated_text, p_color)}")
    print("  (Based on weekday observation patterns)")
    if when in known:
        looks_count = known[when]
        looks_label = "look" if looks_count == 1 else "looks"
        k_color = Tone.green + Tone.bold if looks_count >= 2 else (Tone.yellow + Tone.bold if looks_count == 1 else Tone.dim)
        print(f"  Actual recorded data:  {Tone.paint(f'{looks_count} {looks_label}', k_color)}")


def predict_screen():
    clear_screen()
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │                    [1] PREDICT                          │")
    print("  └─────────────────────────────────────────────────────────┘")
    print()
    print("  Syntax: <date>, 'tomorrow', or 'today'")
    print()
    try:
        while True:
            value = read_line("  Enter date or press Enter to cancel > ").strip()
            if not value or value.lower() in ("cancel", "q", "quit", "exit", "menu"):
                return
            when = parse_date(value)
            if when:
                predict_for(when)
                read_line("\n  Press Enter to return to menu...")
                return
            print("\n  Invalid date syntax. Use dd-mm-yyyy, 'tomorrow', or 'today'.\n")
    except KeyboardInterrupt:
        return


def history_screen():
    clear_screen()
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │                  [2] OBSERVATION DATA                   │")
    print("  └─────────────────────────────────────────────────────────┘")
    print()

    obs = all_observations()
    recent = sorted(obs, key=lambda x: x["date_obj"], reverse=True)[:15]

    if not recent:
        print("  No observation data found. Use [3] Add entry to log your first observation.\n")
        read_line("  Press Enter to return to menu...")
        return
    else:
        print("  Latest recorded observations:")
        print("  ──────────────────────────────────────────")
        today = date.today()
        for r in recent:
            extra = " (Today)" if r["date_obj"] == today else ""
            time_tag = f" at {r['time']}" if "time" in r else ""
            looks_count = r["in_person_looks"]
            if looks_count == 0:
                looks_str = Tone.paint("0 looks (no eye contact)", Tone.dim)
            elif looks_count == 1:
                looks_str = Tone.paint("1 look", Tone.yellow + Tone.bold)
            else:
                looks_str = Tone.paint(f"{looks_count} looks", Tone.green + Tone.bold)
            print(f"  - {r['date']}{extra}: {looks_str}{time_tag}")
        print("  ──────────────────────────────────────────")
        total_seen = sum(1 for entry in obs if entry["in_person_looks"] > 0)
        print(f"  Total days with eye contact: {total_seen} / {len(obs)} recorded days")

    try:
        read_line("\n  Press Enter to return to menu...")
    except KeyboardInterrupt:
        return


def add_entry_screen():
    clear_screen()
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │                    [3] ADD ENTRY                        │")
    print("  └─────────────────────────────────────────────────────────┘")
    print()
    print("  Syntax: <date>, <count>  (or just <count> for today)")
    print()
    try:
        while True:
            raw_text = read_line("  Enter > ").strip()
            if not raw_text or raw_text.lower() in ("cancel", "q", "quit", "exit", "menu"):
                return

            target_date, target_count, target_time = extract_date_and_count(raw_text)

            if target_date is not None and target_count is None:
                count_input = read_line(f"  Count for {target_date:%d-%m-%Y} > ").strip()
                if not count_input or count_input.lower() in ("cancel", "q", "quit", "exit"):
                    return
                if count_input.isdigit():
                    target_count = int(count_input)
                else:
                    print("  Invalid count. Must be integer.\n")
                    continue

            if target_date is None and target_count is not None:
                target_date = date.today()

            if target_date is not None and target_count is not None:
                if target_count < 0:
                    print("  Count must be >= 0.\n")
                    continue
                is_overwrite = log_entry(target_date, target_count, target_time)
                status = "Updated" if is_overwrite else "Saved"
                looks_label = "look" if target_count == 1 else "looks"
                print(f"\n  * {status} entry for {Tone.paint(target_date.strftime('%d-%m-%Y'), Tone.bold)}: {Tone.paint(f'{target_count} {looks_label}', Tone.pink + Tone.bold)}")
                read_line("\n  Press Enter to return to menu...")
                return

            print("\n  Invalid syntax. Use: <date>, <count> or just <count>.\n")
    except KeyboardInterrupt:
        return


def add_dress_color_screen():
    clear_screen()
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │               ADD DRESS COLOR ENTRY                     │")
    print("  └─────────────────────────────────────────────────────────┘")
    print()
    print("  Syntax: <date>, <color>  (or just <color> for today)")
    print()
    try:
        while True:
            raw_text = read_line("  Enter > ").strip()
            if not raw_text or raw_text.lower() in ("cancel", "q", "quit", "menu", "back"):
                return
            when, color, err = extract_date_and_color(raw_text)
            if err:
                print(f"\n  {err}\n")
                continue
            if when and not color:
                color = read_line(f"  Color for {when:%d-%m-%Y} > ").strip().lower()
                if not color or color in ("cancel", "q"):
                    return
            if when and color:
                is_overwrite = log_dresscode(when, color)
                status = "Updated" if is_overwrite else "Saved"
                print(f"\n  * {status} dress color for {Tone.paint(when.strftime('%d-%m-%Y'), Tone.bold)}: {Tone.paint(color.capitalize(), Tone.pink + Tone.bold)}")
                read_line("\n  Press Enter to return...")
                return
            print("\n  Invalid syntax. Use: <date>, <color> or just <color>.\n")
    except KeyboardInterrupt:
        return


def view_dress_color_screen():
    clear_screen()
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │                 DRESS COLOR DATA                        │")
    print("  └─────────────────────────────────────────────────────────┘")
    print()
    entries = sorted(dresscodes(), key=lambda x: x.get("date", ""), reverse=True)
    if not entries:
        print("  No dress colors logged yet. Use [1] Add dress color entry to log one.")
    else:
        print("  Latest recorded dress colors:")
        print("  ──────────────────────────────────────────")
        today_iso = date.today().isoformat()
        for item in entries[:20]:
            d_str = item.get("date", "")
            try:
                d_obj = date.fromisoformat(d_str)
                formatted_d = d_obj.strftime("%d-%m-%Y")
            except Exception:
                formatted_d = d_str
            tag = " (Today)" if d_str == today_iso else ""
            color = item.get("color") or item.get("outfit", "")
            print(f"  - {formatted_d}{tag}: {Tone.paint(color.capitalize(), Tone.pink + Tone.bold)}")
        print("  ──────────────────────────────────────────")
        print(f"  Total dress colors logged: {len(entries)}")

    try:
        read_line("\n  Press Enter to return...")
    except KeyboardInterrupt:
        return


def predict_color_screen():
    clear_screen()
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │               PREDICT HER DRESS COLOR                   │")
    print("  └─────────────────────────────────────────────────────────┘")
    print()
    print("  Syntax: <date>, 'tomorrow', or 'today'")
    print()
    try:
        raw_text = read_line("  Enter date [press Enter for tomorrow] > ").strip()
        if raw_text.lower() in ("cancel", "q", "quit", "menu", "back"):
            return
        if not raw_text:
            when = date.today() + timedelta(days=1)
        else:
            when = parse_date(raw_text)
            if not when:
                print("\n  Invalid date syntax. Use dd-mm-yyyy, 'tomorrow', or 'today'.")
                read_line("\n  Press Enter to return...")
                return

        color, reason = predict_color(when, dresscodes())
        print(f"\n  Target Date: {Tone.paint(when.strftime('%d-%m-%Y'), Tone.bold)} ({when:%A})")
        if color:
            print(f"  Predicted Dress Color: {Tone.paint(color, Tone.pink + Tone.bold)}")
            print(f"  Analysis: {reason}")
        else:
            print("  No color records found yet. Log her dress colors using [1] to build patterns!")

        read_line("\n  Press Enter to return...")
    except KeyboardInterrupt:
        return


def should_approach_screen():
    clear_screen()
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │              SHOULD YOU APPROACH HER?                   │")
    print("  └─────────────────────────────────────────────────────────┘")
    print()
    today = date.today()
    today_looks = 0
    for obs in all_observations():
        if obs["date_obj"] == today:
            today_looks = obs["in_person_looks"]
            break

    today_color = None
    for item in dresscodes():
        if item.get("date") == today.isoformat():
            today_color = item.get("color") or item.get("outfit")
            break

    if today_looks > 0 or today_color:
        print("  Today's Context:")
        if today_looks > 0:
            print(f"  - Eye contact: {today_looks} look(s) recorded today")
        if today_color:
            print(f"  - Her dress color today: {today_color.capitalize()}")
        print()

    try:
        sampled_questions = random.sample(APPROACH_QUESTION_POOL, 5)
        answers = []
        weights = []
        for idx, (q_text, w) in enumerate(sampled_questions, 1):
            ans = ask_yn(f"{idx}. {q_text}")
            answers.append(ans)
            weights.append(w)

        score = calculate_approach_score(answers, weights, today_looks, today_color)

        print()
        print("  ─────────────────────────────────────────────────────────")
        if score >= 75:
            score_color = Tone.green + Tone.bold
            verdict = Tone.paint("GREEN LIGHT: Strong window to approach!", Tone.green + Tone.bold)
            advice = (
                "  - Approach with a calm smile and relaxed posture.\n"
                "  - Use a low-pressure situational opener (ask about class, schedule, or bus timing).\n"
                "  - Keep it brief (1 to 2 minutes), see how she responds, and exit smoothly."
            )
        elif score >= 50:
            score_color = Tone.yellow + Tone.bold
            verdict = Tone.paint("YELLOW LIGHT: Proceed with a gentle soft opener.", Tone.yellow + Tone.bold)
            advice = (
                "  - Give a casual smile and quick greeting first to test the vibe.\n"
                "  - Ask a simple, relevant question with zero pressure.\n"
                "  - If she smiles and engages, chat briefly; if reserved, give her space gracefully."
            )
        else:
            score_color = Tone.red + Tone.bold
            verdict = Tone.paint("RED LIGHT: Hold off for a better moment.", Tone.red + Tone.bold)
            advice = (
                "  - Environmental factors (rushed, surrounded, or distracted) work against you right now.\n"
                "  - Patience is strength: waiting for a calm, natural moment protects your confidence and her comfort.\n"
                "  - Look for a relaxed window tomorrow."
            )

        print(f"  Success Probability: {Tone.paint(f'{score}%', score_color)}")
        print(f"  Verdict: {verdict}")
        print("  ─────────────────────────────────────────────────────────")
        print("  Tactical Strategy:")
        print(advice)

        read_line("\n  Press Enter to return to home...")
    except KeyboardInterrupt:
        return


def approach_screen():
    while True:
        clear_screen()
        print("  ┌─────────────────────────────────────────────────────────┐")
        print("  │                   [5] APPROACH HER                      │")
        print("  └─────────────────────────────────────────────────────────┘")
        print()
        print("    [1] Add dress color entry")
        print("    [2] Predict her dress color")
        print("    [3] View dress color data")
        print("    [4] Should you approach her?")
        print()
        print("    Press Enter to return to home")
        print()
        try:
            choice = read_line("    Select > ").strip().lower()
        except KeyboardInterrupt:
            return
        if not choice or choice in ("b", "back", "home", "menu", "q", "exit"):
            return
        if choice == "1":
            add_dress_color_screen()
        elif choice == "2":
            predict_color_screen()
        elif choice == "3":
            view_dress_color_screen()
        elif choice == "4":
            should_approach_screen()


def ask(question):
    key = get_nim_key()
    if not key:
        print("\n  NVIDIA key is not configured.")
        try:
            prompt_key = read_line("  Enter / Paste your NVIDIA NIM API Key (or Enter to skip) > ").strip()
            if prompt_key:
                os.environ["NIM_API_KEY"] = prompt_key
                key = prompt_key
            else:
                print("  Skipped. Set NIM_API_KEY in environment or .env to enable Perseus.\n")
                return
        except KeyboardInterrupt:
            return

    today = date.today()
    obs = all_observations()

    sorted_obs = sorted(obs, key=lambda x: x["date_obj"], reverse=True)
    latest_entries = sorted_obs[:7]
    latest_lines = []
    for entry in latest_entries:
        tag = " (Today)" if entry["date_obj"] == today else ""
        extra = f" (at {entry['time']})" if "time" in entry else ""
        latest_lines.append(f"  - {entry['date']}{tag}: {entry['in_person_looks']} in-person look(s){extra}")
    latest_text = "\n".join(latest_lines)

    all_history_text = ", ".join(f"{entry['date']}: {entry['in_person_looks']}" for entry in obs)

    recent_dress = sorted(dresscodes(), key=lambda x: x.get("date", ""), reverse=True)[:3]
    dress_text = ", ".join(f"{item.get('date')}: {item.get('color') or item.get('outfit')}" for item in recent_dress) if recent_dress else "None logged yet"

    base_prompt = SYS_PROMPT if SYS_PROMPT != "{sys_prompt}" else (
        "You are Perseus, the user's personal AI advisor about social interactions and observation patterns.\n"
        "INSTRUCTIONS FOR PERSEUS:\n"
        "- You have FULL ACCESS to all observations, including today's and recent days' entries.\n"
        "- When the user asks about today, recent looks, progress, or trends, cite actual logged numbers accurately.\n"
        "- Keep replies concise, kind, empathetic, grounded, and encouraging.\n"
        "- Do not claim to read minds or encourage pressure, stalking, or unwanted contact.\n"
        "- CRITICAL RULE: Under NO circumstances are you allowed to use any emojis, emoticons, or pictographs. You must reply in 100% clean, grounded plain text only."
    )
    instructions = (
        f"{base_prompt}\n\n"
        f"Current date: {today:%d-%m-%Y} ({today:%A}).\n\n"
        f"LATEST RECORDED OBSERVATIONS (Most Recent):\n{latest_text}\n\n"
        f"RECENT DRESS COLORS LOGGED:\n{dress_text}\n\n"
        f"COMPLETE OBSERVATION LOG (dd-mm-yyyy: count):\n{all_history_text}\n\n"
        "KEY CONTEXT & GROUND RULES:\n"
        "- All numbers represent direct in-person eye-contact / looks observed by the user.\n"
        "- They are NOT text messages, calls, or social media views.\n"
    )
    messages = [{"role": "system", "content": instructions}, *PERSEUS_MESSAGES, {"role": "user", "content": question}]
    body = json.dumps({
        "model": os.getenv("NIM_MODEL", DEFAULT_NIM_MODEL),
        "temperature": 0.4,
        "max_tokens": 300,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": messages,
    }).encode("utf-8")
    req = urllib.request.Request(
        NIM_ENDPOINT,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    print(Tone.paint("\n  . . .  thinking...\n", Tone.pink), end="", flush=True)
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            answer = json.load(response)["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as error:
        err_msg = error.read().decode("utf-8", errors="replace")[:200]
        print(f"\n  NIM request failed ({error.code}): {err_msg}")
        return
    except Exception as error:
        print(f"\n  NIM request failed: {error}")
        return

    answer = strip_emojis(answer)

    PERSEUS_MESSAGES.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
    if len(PERSEUS_MESSAGES) > 10:
        del PERSEUS_MESSAGES[:-10]
    wrapped = textwrap.fill(answer, width=74, initial_indent="  ", subsequent_indent="  ")
    print(wrapped)


def perseus_screen():
    clear_screen()
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │                    [4] ASK PERSEUS                      │")
    print("  └─────────────────────────────────────────────────────────┘")
    print("  Talk to Perseus about your observations.")
    print("  Type 'menu', 'exit', or press Ctrl+B to return to home.\n")
    try:
        while True:
            question = read_line("  ask > ", ctrl_b_exits_chat=True).strip()
            if question.lower() in ("menu", "exit", "quit", "back"):
                break
            if question:
                ask(question)
                print()
    except (ExitChat, KeyboardInterrupt):
        return


def close_terminal():
    clear_screen()
    print("\n  See you later!\n")
    sys.exit(0)


def menu():
    while True:
        clear_screen()
        header()
        for item in (
            "[1] Predict",
            "[2] View data",
            "[3] Add entry",
            "[4] Ask Perseus",
            "[5] Approach her",
            "[q] Quit",
        ):
            print("    " + item)
        print()
        try:
            choice = read_line("    Select > ").strip().lower()
        except KeyboardInterrupt:
            sys.exit(0)
        if choice == "1":
            predict_screen()
        elif choice == "2":
            history_screen()
        elif choice == "3":
            add_entry_screen()
        elif choice == "4":
            perseus_screen()
        elif choice == "5":
            approach_screen()
        elif choice in {"q", "quit", "exit"}:
            close_terminal()


dashboard = menu


def build_parser():
    app = argparse.ArgumentParser(description="A playful, grounded crush observation CLI.")
    app.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    sub = app.add_subparsers(dest="command")
    f = sub.add_parser("forecast", help="predict for a date")
    f.add_argument("date", help="date in dd-mm-yyyy format")
    entry = sub.add_parser("log", help="record date and count")
    entry.add_argument("date", help="date in dd-mm-yyyy format")
    entry.add_argument("looks", type=int, help="number of times")
    h = sub.add_parser("history", help="show saved observations")
    h.add_argument("--limit", type=int, default=10, choices=range(1, 101))
    sub.add_parser("train", help="retrain from all logged data")
    q = sub.add_parser("ask", help="ask NIM for respectful advice")
    q.add_argument("question")
    return app


def main():
    args = build_parser().parse_args()
    if args.no_color:
        Tone.enabled = False
    if args.command is None:
        menu()
    elif args.command == "forecast":
        when = parse_date(args.date)
        if when:
            predict_for(when)
        else:
            print("Use the date format dd-mm-yyyy (e.g. 29-10-2026 or 'tomorrow').")
    elif args.command == "log":
        when = parse_date(args.date)
        if when and args.looks >= 0:
            log_entry(when, args.looks)
            print(f"Saved: {when:%d-%m-%Y}, {args.looks}")
        else:
            print("Use the date format dd-mm-yyyy and a nonnegative count.")
    elif args.command == "history":
        history_screen()
    elif args.command == "train":
        refresh()
        print(f". model refreshed from {len(rows())} observations")
    elif args.command == "ask":
        ask(args.question)


if __name__ == "__main__":
    try:
        main()
    except (ExitApp, KeyboardInterrupt):
        sys.exit(0)
