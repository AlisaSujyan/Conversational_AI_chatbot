import logging
import re
import sqlite3
import sys
from io import StringIO
import os
import asyncio
from functools import partial

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
import requests
import wikipedia

try:
    from PIL import Image
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8425721052:AAEXhnf1PA-yab9AN4wHM3mKi25YsrlANwM")

DB_PATH = "assistant_v14.db"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_HISTORY = 4
MAX_SUMMARY_CHARS = 8000
MAX_RESPONSE_TOKENS = 250
CODE_MAX_RESPONSE_TOKENS = 600

model = None
tokenizer = None
response_cache = {}

PERSONALITY_MODES = {
    "teacher": {
        "name": "Teacher",
        "emoji": "👨‍🏫",
        "prompt": (
            "Act as a helpful teacher who explains things clearly with examples. "
            "Do not repeat this description or refer to yourself as a mode."
        ),
    },
    "friend": {
        "name": "Friend",
        "emoji": "👋",
        "prompt": (
            "Act as a friendly assistant who gives brief, helpful answers. "
            "Do not repeat this description or say 'I am a friendly assistant'."
        ),
    },
    "expert": {
        "name": "Expert",
        "emoji": "🔬",
        "prompt": (
            "Act as a technical expert who provides precise, detailed information. "
            "Do not repeat this description in your reply."
        ),
    },
    "coder": {
        "name": "Coder",
        "emoji": "💻",
        "prompt": (
            "Act as a coding assistant. When the user asks for code, respond with a complete "
            "solution in a single ```python``` block first, then optionally a short explanation. "
            "Do not repeat this description or introduce yourself; just output the answer."
        ),
    },
    "search": {
        "name": "Search",
        "emoji": "🔍",
        "prompt": (
            "Act as an information specialist who provides well-organized, factual answers. "
            "Do not repeat this description in your reply."
        ),
    },
}

CLARIFYING_ADDITION = " Ask a brief clarifying question if needed."


# ==================== DATABASE FUNCTIONS ====================

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.executescript(
        """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    mode TEXT NOT NULL DEFAULT 'friend',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    user_msg TEXT NOT NULL,
    bot_msg TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS bookmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    preview TEXT NOT NULL,
    full_response TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stats (
    user_id INTEGER PRIMARY KEY,
    messages INTEGER NOT NULL DEFAULT 0,
    calculations INTEGER NOT NULL DEFAULT 0,
    searches INTEGER NOT NULL DEFAULT 0,
    code_executions INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS mode_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    mode TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    UNIQUE(user_id, mode)
);

CREATE TABLE IF NOT EXISTS reactions (
    user_id INTEGER PRIMARY KEY,
    good INTEGER NOT NULL DEFAULT 0,
    bad INTEGER NOT NULL DEFAULT 0,
    confused INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    file_type TEXT NOT NULL,
    path TEXT NOT NULL,
    name TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""
    )
    conn.commit()
    conn.close()


def get_user_mode(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT mode FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    if row:
        mode = row["mode"]
    else:
        mode = "friend"
        cur.execute("INSERT INTO users (user_id, mode) VALUES (?, ?)", (user_id, mode))
        conn.commit()
    conn.close()
    return mode


def set_user_mode(user_id, mode):
    if mode not in PERSONALITY_MODES:
        return
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO users (user_id, mode) VALUES (?, ?)", (user_id, mode))
    cur.execute("UPDATE users SET mode=? WHERE user_id=?", (mode, user_id))
    conn.commit()
    conn.close()
    increment_mode_usage(user_id, mode)


def add_exchange(user_id, user_msg, bot_msg):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO memory (user_id, user_msg, bot_msg) VALUES (?, ?, ?)",
        (user_id, user_msg, bot_msg),
    )
    conn.commit()
    conn.close()
    trim_history(user_id)


def trim_history(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM memory WHERE user_id=? ORDER BY created_at DESC, id DESC LIMIT ?",
        (user_id, MAX_HISTORY),
    )
    rows = cur.fetchall()
    if not rows:
        conn.close()
        return
    keep_ids = [str(r["id"]) for r in rows]
    placeholders = ",".join("?" for _ in keep_ids)
    cur.execute(
        f"DELETE FROM memory WHERE user_id=? AND id NOT IN ({placeholders})",
        (user_id, *keep_ids),
    )
    conn.commit()
    conn.close()


def get_history(user_id, limit=4):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT user_msg, bot_msg FROM memory WHERE user_id=? ORDER BY created_at ASC, id ASC",
        (user_id,),
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return []
    return rows[-limit:]


def clear_history_db(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM memory WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()


def ensure_stats(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO stats (user_id) VALUES (?)", (user_id,))
    conn.commit()
    conn.close()


def increment_stat(user_id, field):
    if field not in ["messages", "calculations", "searches", "code_executions"]:
        return
    ensure_stats(user_id)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        f"UPDATE stats SET {field} = {field} + 1 WHERE user_id=?",
        (user_id,),
    )
    conn.commit()
    conn.close()


def increment_mode_usage(user_id, mode):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO mode_usage (user_id, mode, count) VALUES (?, ?, 0)",
        (user_id, mode),
    )
    cur.execute(
        "UPDATE mode_usage SET count = count + 1 WHERE user_id=? AND mode=?",
        (user_id, mode),
    )
    conn.commit()
    conn.close()


def get_stats(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM stats WHERE user_id=?", (user_id,))
    srow = cur.fetchone()
    if not srow:
        sdata = {"messages": 0, "calculations": 0, "searches": 0, "code_executions": 0}
    else:
        sdata = {
            "messages": srow["messages"],
            "calculations": srow["calculations"],
            "searches": srow["searches"],
            "code_executions": srow["code_executions"],
        }
    cur.execute("SELECT mode, count FROM mode_usage WHERE user_id=?", (user_id,))
    modes = cur.fetchall()
    mode_counts = {r["mode"]: r["count"] for r in modes}
    conn.close()
    return sdata, mode_counts


def clear_stats(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM stats WHERE user_id=?", (user_id,))
    cur.execute("DELETE FROM mode_usage WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()


def get_reactions(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT good, bad, confused FROM reactions WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return {"good": 0, "bad": 0, "confused": 0}
    return {"good": row["good"], "bad": row["bad"], "confused": row["confused"]}


def increment_reaction(user_id, kind):
    if kind not in ["good", "bad", "confused"]:
        return
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO reactions (user_id, good, bad, confused) VALUES (?, 0, 0, 0)",
        (user_id,),
    )
    cur.execute(
        f"UPDATE reactions SET {kind} = {kind} + 1 WHERE user_id=?",
        (user_id,),
    )
    conn.commit()
    conn.close()


def clear_reactions(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM reactions WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()


def add_bookmark(user_id, full_response):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM bookmarks WHERE user_id=?", (user_id,))
    c = cur.fetchone()["c"]
    if c >= 20:
        conn.close()
        return False
    preview = full_response[:100]
    cur.execute(
        "INSERT INTO bookmarks (user_id, preview, full_response) VALUES (?, ?, ?)",
        (user_id, preview, full_response),
    )
    conn.commit()
    conn.close()
    return True


def get_bookmarks_db(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, preview FROM bookmarks WHERE user_id=? ORDER BY created_at DESC, id DESC",
        (user_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def clear_bookmarks_db(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM bookmarks WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()


def add_file_record(user_id, file_type, path, name):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO files (user_id, file_type, path, name) VALUES (?, ?, ?, ?)",
        (user_id, file_type, path, name),
    )
    conn.commit()
    conn.close()


# ==================== UTILITY FUNCTIONS ====================

def history_to_text(history_rows):
    """Cleaner history format (currently unused, kept for flexibility)."""
    if not history_rows:
        return ""
    lines = []
    for row in history_rows[-3:]:
        user_msg = row["user_msg"][:100]
        bot_msg = row["bot_msg"][:100]
        lines.append(f"User: {user_msg}")
        lines.append(f"Assistant: {bot_msg}")
    return "\n".join(lines)


def strip_markdown_for_fallback(text):
    return text.replace("**", "").replace("*", "").replace("_", "").replace("`", "")


def format_code_response(response):
    # If there is already a fenced block, don't touch it
    if "```" in response:
        return response

    # Common pattern: model outputs a line that is literally "python"
    lines = response.splitlines()
    if lines and lines[0].strip().lower() == "python":
        lines = lines[1:]
        response = "\n".join(lines).lstrip()

    # Heuristic: detect if this looks like code
    code_pattern = r"(def |class |import |for |while |if |return )"
    if re.search(code_pattern, response):
        return f"```python\n{response}\n```"

    return response


# ==================== MATH & DERIVATIVE FUNCTIONS ====================

def extract_equation_from_text(text):
    t = text.replace(" ", "")
    pattern = r"[0-9]*[+-]?[0-9]*x\^?\d*([+-][0-9]*x\^?\d*)*([+-]\d+)?"
    m = re.search(pattern, t)
    if m:
        return m.group()
    return None


def parse_polynomial(equation):
    eq = equation.replace(" ", "")
    eq = eq.replace("-", "+-")
    if eq.startswith("+-"):
        eq = eq[1:]
    terms = eq.split("+")
    coeffs = {}
    for t in terms:
        if not t:
            continue
        if "x" not in t:
            c = float(t)
            p = 0
        else:
            if "^" in t:
                cpart, ppart = t.split("x^")
                p = int(ppart)
            else:
                cpart = t.split("x")[0]
                p = 1
            if cpart in ["", "+"]:
                c = 1.0
            elif cpart == "-":
                c = -1.0
            else:
                c = float(cpart)
        coeffs[p] = coeffs.get(p, 0.0) + c
    return coeffs


def derivative_polynomial(equation):
    coeffs = parse_polynomial(equation)
    deriv = {}
    for p, c in coeffs.items():
        if p == 0:
            continue
        deriv[p - 1] = deriv.get(p - 1, 0.0) + c * p
    parts = []
    for p in sorted(deriv.keys(), reverse=True):
        c = deriv[p]
        if abs(c) < 1e-9:
            continue
        sign = "+" if c > 0 else "-"
        val = abs(c)
        if p == 0:
            part = f"{val:.2f}"
        elif p == 1:
            part = "x" if val == 1 else f"{val:.2f}x"
        else:
            part = f"x^{p}" if val == 1 else f"{val:.2f}x^{p}"
        parts.append((sign, part))
    if not parts:
        return "0"
    fs, fp = parts[0]
    expr = ("-" if fs == "-" else "") + fp
    for s, p in parts[1:]:
        expr += f" {s} {p}"
    return expr


def calculate_basic_math(text, user_id):
    """FAST math calculation"""
    result = {"has_calc": False, "question": "", "answer": ""}

    try:
        lower = text.lower()

        if any(word in lower for word in
               ["derivative", "differentiate", "roots", "solve", "equation", "quadratic", "complex"]):
            return result

        m = re.search(r"(\d+\.?\d*)\s*%\s*of\s*(\d+\.?\d*)", text, re.IGNORECASE)
        if m:
            percent = float(m.group(1))
            number = float(m.group(2))
            answer = number * (percent / 100)
            result["has_calc"] = True
            result["question"] = f"{percent}% of {number}"
            result["answer"] = f"{answer:.2f}"
            increment_stat(user_id, "calculations")
            return result

        m = re.search(r"(\d+\.?\d*)\s*%\s*tip\s*(?:on\s*)?\$?(\d+\.?\d*)", text, re.IGNORECASE)
        if m:
            percent = float(m.group(1))
            amount = float(m.group(2))
            tip = amount * (percent / 100)
            total = amount + tip
            result["has_calc"] = True
            result["question"] = f"{percent}% tip on ${amount}"
            result["answer"] = f"Tip: ${tip:.2f}, Total: ${total:.2f}"
            increment_stat(user_id, "calculations")
            return result

        if "x" not in lower and "y" not in lower and "z" not in lower:
            m = re.search(r"(\d+\.?\d*)\s*([+\-*/×÷])\s*(\d+\.?\d*)", text)
            if m:
                num1 = float(m.group(1))
                op = m.group(2).replace("×", "*").replace("÷", "/")
                num2 = float(m.group(3))

                if op == "+":
                    answer = num1 + num2
                elif op == "-":
                    answer = num1 - num2
                elif op == "*":
                    answer = num1 * num2
                elif op == "/":
                    answer = num1 / num2 if num2 != 0 else "Error: Division by zero"
                else:
                    return result

                if isinstance(answer, (int, float)):
                    result["has_calc"] = True
                    result["question"] = f"{num1} {op} {num2}"
                    result["answer"] = f"{answer:.2f}"
                    increment_stat(user_id, "calculations")
                    return result

        return result
    except Exception:
        return result


def handle_derivative_request(text, user_id, history_rows):
    """Handle derivative calculations"""
    lower = text.lower()
    if "derivative" not in lower and "differentiate" not in lower:
        return None
    eq = extract_equation_from_text(text)
    if not eq:
        for row in reversed(history_rows):
            e = extract_equation_from_text(row["user_msg"])
            if e:
                eq = e
                break
    if not eq:
        return "I couldn't detect an equation. Please provide it like: x^2 + 3x + 5"
    deriv = derivative_polynomial(eq)
    increment_stat(user_id, "calculations")
    return f"📐 The derivative of {eq} is:\n\n{deriv}"


# ==================== CODE EXECUTION ====================

FORBIDDEN_CODE = [
    "import ", "import\t", "open(", "exec(", "eval(",
    "__", "os.", "sys.", "subprocess", "socket",
    "shutil", "pickle", "input(",
]


def is_safe_code(code):
    low = code.lower()
    return not any(f in low for f in FORBIDDEN_CODE)


def execute_code(code, user_id):
    """Safe code execution"""
    result = {"success": False, "output": "", "error": ""}

    if not is_safe_code(code):
        result["error"] = "⚠️ This code is not allowed for safety reasons."
        return result

    old_stdout = sys.stdout
    try:
        sys.stdout = StringIO()

        exec_globals = {
            "__builtins__": {
                "print": print, "len": len, "range": range, "sum": sum,
                "max": max, "min": min, "abs": abs, "round": round,
                "int": int, "float": float, "str": str, "list": list,
                "dict": dict, "set": set, "enumerate": enumerate,
                "zip": zip, "sorted": sorted,
            }
        }

        exec(code, exec_globals)
        output = sys.stdout.getvalue()


        result["success"] = True
        result["output"] = output if output else "✓ Code executed successfully (no output)"
        return result

    except Exception as e:
        result["error"] = str(e)
        return result

    finally:
        sys.stdout = old_stdout

# ==================== SEARCH FUNCTIONS ====================

def search_wikipedia(query, user_id):
    """Fast Wikipedia search with context tweaks"""
    result = {"success": False, "title": "", "summary": "", "url": ""}

    try:
        query = re.sub(r"(who invented|who created|who is|what is|tell me about)",
                       "", query, flags=re.IGNORECASE).strip()

        if "python" in query.lower() and any(
                kw in query.lower() for kw in ["invented", "created", "programming", "language", "guido"]):
            query = "Python programming language"

        search_results = wikipedia.search(query, results=5)
        if not search_results:
            return result

        best_match = search_results[0]
        if "python" in query.lower():
            for r in search_results:
                if "programming" in r.lower() or "language" in r.lower() or "guido" in r.lower():
                    best_match = r
                    break

        page = wikipedia.page(best_match, auto_suggest=False)
        result["success"] = True
        result["title"] = page.title
        result["summary"] = page.summary[:500]
        result["url"] = page.url
        increment_stat(user_id, "searches")
        return result
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            page = wikipedia.page(e.options[0], auto_suggest=False)
            result["success"] = True
            result["title"] = page.title
            result["summary"] = page.summary[:500]
            result["url"] = page.url
            increment_stat(user_id, "searches")
        except Exception:
            pass
    except Exception:
        pass
    return result


def search_web(query, user_id):
    """DuckDuckGo search"""
    result = {"success": False, "title": "", "content": "", "source": ""}

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://api.duckduckgo.com/?q={query}&format=json&no_redirect=1"
        response = requests.get(url, headers=headers, timeout=5)

        if response.status_code == 200:
            data = response.json()
            if data.get("AbstractText"):
                result["success"] = True
                result["title"] = data.get("Heading", "Search Result")
                result["content"] = data["AbstractText"][:600]
                result["source"] = data.get("AbstractSource", "DuckDuckGo")
                increment_stat(user_id, "searches")
        return result
    except Exception:
        return result


# ==================== FILE PROCESSING ====================

def extract_text_from_image(path, user_id):
    """OCR for images"""
    if not OCR_AVAILABLE:
        return ("❌ OCR not available.\n\n"
                "To enable text recognition from images, install:\n"
                "  pip install pytesseract pillow\n"
                "and make sure the Tesseract OCR binary is installed on the system.")

    try:
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        increment_stat(user_id, "searches")
        if not text.strip():
            return "❌ No readable text found in this image."
        return text
    except pytesseract.TesseractNotFoundError:
        return ("❌ Tesseract is not installed or not in PATH.\n\n"
                "Install the Tesseract OCR program on your system, "
                "then restart the bot.")
    except Exception as e:
        return f"❌ Error processing image: {str(e)}"


def extract_text_from_pdf(path, user_id):
    """PDF text extraction"""
    if not PDF_AVAILABLE:
        return "PDF processing not available. Install: pip install PyPDF2"
    try:
        text = ""
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        text = text.strip()
        increment_stat(user_id, "searches")
        if not text:
            return "❌ No text extracted from PDF."
        if len(text) > MAX_SUMMARY_CHARS:
            text = text[:MAX_SUMMARY_CHARS]
        return text
    except Exception as e:
        return f"❌ Error processing PDF: {str(e)}"


# ==================== TOOL DETECTION ====================

def detect_tool_need(text: str, mode: str) -> dict:
    tool_need = {"type": None, "query": None}
    text_lower = text.lower()

    # Let simple arithmetic be handled by calculate_basic_math
    if re.search(r"\d+\.?\d*\s*[+\-*/×÷%]\s*\d+\.?\d*", text) and "x" not in text_lower:
        return tool_need

    # Explicit "run this code" execution
    if ("run this" in text_lower or "execute this" in text_lower) and "```" in text:
        tool_need["type"] = "execute_code"
        return tool_need

    code_keywords = ["write code", "python code", "code for", "write a function", "create function"]
    creator_keywords = ["invented", "created", "founder", "author", "discovered"]
    wikipedia_keywords = ["who", "what is", "when was", "where", "history of", "tell me about"]
    web_keywords = ["search for", "look up", "find information", "current", "latest", "recent"]
    explicit_wikipedia_triggers = ["wikipedia", "wiki"]

    # Don't trigger tools on basic meta-questions about the bot itself
    if "who are you" in text_lower or ("what can you do" in text_lower and "you" in text_lower):
        return tool_need

    # User explicitly asking for code style assistance
    if any(kw in text_lower for kw in code_keywords):
        tool_need["type"] = "code"
        return tool_need

    # Generic web search
    if any(kw in text_lower for kw in web_keywords):
        tool_need["type"] = "web"
        tool_need["query"] = text
        return tool_need

    # Wikipedia behaviour depends on mode
    if mode == "search":
        if (
            any(kw in text_lower for kw in creator_keywords)
            or any(kw in text_lower for kw in wikipedia_keywords)
            or any(kw in text_lower for kw in explicit_wikipedia_triggers)
        ):
            tool_need["type"] = "wikipedia"
            tool_need["query"] = text
            return tool_need
    else:
        if any(kw in text_lower for kw in explicit_wikipedia_triggers):
            tool_need["type"] = "wikipedia"
            tool_need["query"] = text
            return tool_need

    return tool_need


def detect_ambiguous_question(text, history_rows):
    """Detect vague questions"""
    ambiguous_words = ["which", "what one", "that", "it", "this", "better", "should i"]
    has_ambiguous = any(word in text.lower() for word in ambiguous_words)
    is_very_short = len(text.split()) <= 4
    no_history = len(history_rows) == 0
    return has_ambiguous and is_very_short and no_history


def build_feedback_prompt(user_id):
    """Adapt based on user reactions"""
    reac = get_reactions(user_id)
    good = reac.get("good", 0)
    bad = reac.get("bad", 0)
    confused = reac.get("confused", 0)
    extra = ""
    if bad > good:
        extra += " Be extra clear and accurate."
    if confused > good:
        extra += " Explain step-by-step."
    return extra


def is_code_request(user_message, mode):
    if not user_message:
        return False

    # 🚫 Internal tool prompts are never "code requests"
    if user_message.startswith("Summarize this PDF content concisely") \
       or user_message.startswith("Analyze this text extracted from an image"):
        return False

    # In normal chat, coder mode prefers code-style answers
    if mode == "coder":
        return True

    keywords = ["code", "python", "function", "class", "script"]
    return any(k in user_message.lower() for k in keywords)


# ==================== MODEL LOADING ====================

def ensure_model_loaded():
    """Load model once"""
    global model, tokenizer
    if model is not None and tokenizer is not None:
        return

    logger.info("⏳ Loading Mistral-7B (optimized)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    mdl.eval()

    tokenizer = tok
    model = mdl
    logger.info("✓ Model loaded!")


# ==================== RESPONSE GENERATION (FIXED!) ====================

def clean_response(response):
    """Remove system prompt artifacts and instruction leakage as aggressively as possible."""
    # Remove common instruction artifacts
    artifacts = [
        "[INST]", "[/INST]", "Your role:", "Your style:", "Your approach:",
        "Your focus:", "Your task:", "When explaining", "Summarize this", "Note:",
        # Explicitly strip leaked PDF instruction lines if they appear
        "Summarize this PDF content concisely in your own words. Do not repeat the full text and do not mention these instructions. Just output a short summary:",
        "PDF content concisely in your own words. Do not repeat the full text and do not mention these instructions. Just output a short summary:",
    ]
    for artifact in artifacts:
        response = response.replace(artifact, "")

    # Remove personality-mode prompt text if it leaked
    for mode in PERSONALITY_MODES.values():
        prompt_text = mode["prompt"]
        if prompt_text in response:
            response = response.replace(prompt_text, "")

    # Remove generic "You are ... assistant" style lines
    lines = response.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("you are a ") and "assistant" in stripped.lower():
            continue
        if stripped == CLARIFYING_ADDITION.strip():
            continue
        cleaned_lines.append(line)
    response = "\n".join(cleaned_lines)

    # Remove instruction-style bullets that sometimes leak
    lines = response.split("\n")
    final_lines = []
    for line in lines:
        stripped = line.strip()
        if any(pattern in stripped for pattern in [
            "- Be ", "- Use ", "- Provide ", "- Keep ", "- Stay ",
            "- Don't ", "- Explain ", "Be casual", "Be helpful", "Stay on topic"
        ]) and len(stripped) < 120:
            continue
        final_lines.append(line)
    response = "\n".join(final_lines)

    # Remove repeated punctuation
    response = re.sub(r"([.!?])\1+", r"\1", response)

    # Strip excess blank lines
    response = "\n".join(l for l in response.split("\n") if l.strip())

    return response.strip()


def generate_response_sync(user_message, user_id, continue_generation, previous_prompt_cache):
    """Prevents hallucinations and system prompt leakage, supports real continuation and code-aware generation."""
    ensure_model_loaded()
    history_rows = get_history(user_id, limit=MAX_HISTORY)
    current_mode = get_user_mode(user_id)

    # Detect internal tool-style tasks where we DO NOT want chat history
    is_pdf_summary = user_message.startswith("Summarize this PDF content concisely")
    is_ocr_analysis = user_message.startswith("Analyze this text extracted from an image")

    if is_pdf_summary or is_ocr_analysis:
        # For tool prompts, ignore previous conversation completely
        history_rows = []

    # Detect if this interaction is code-focused
    if not continue_generation:
        # For PDF/OCR tool calls, never treat as code
        if is_pdf_summary or is_ocr_analysis:
            is_code = False
        else:
            is_code = is_code_request(user_message, current_mode)
    else:
        previous_answer = user_message or ""
        # We don't currently "continue" PDF/OCR answers, but be safe anyway
        if is_pdf_summary or is_ocr_analysis:
            is_code = False
        else:
            is_code = (
                "```" in previous_answer
                or re.search(r"\bdef\b|\bclass\b|import ", previous_answer)
                or current_mode == "coder"
            )

    # ========== 1) NEW REQUEST (not continuation) ==========
    if not continue_generation:
        # 1. Math first
        math_result = calculate_basic_math(user_message, user_id)
        if math_result["has_calc"]:
            resp = f"🧮 {math_result['question']} = {math_result['answer']}"
            increment_stat(user_id, "messages")
            increment_mode_usage(user_id, current_mode)
            add_exchange(user_id, user_message, resp)
            return resp, False, None, resp

        # 2. Derivatives
        derivative_resp = handle_derivative_request(user_message, user_id, history_rows)
        if derivative_resp is not None:
            increment_stat(user_id, "messages")
            increment_mode_usage(user_id, current_mode)
            add_exchange(user_id, user_message, derivative_resp)
            return derivative_resp, False, None, derivative_resp

        # 3. Tools (mode-aware)
        tool_need = detect_tool_need(user_message, current_mode)

        if tool_need["type"] == "execute_code":
            code_match = re.search(r"```(?:python)?\n(.*?)```", user_message, re.DOTALL)
            if code_match:
                increment_stat(user_id, "code_executions")  # ✅ count run attempt

                code = code_match.group(1)
                exec_result = execute_code(code, user_id)

                if exec_result["success"]:
                    resp = f"💻 Code Output:\n\n{exec_result['output']}"
                else:
                    resp = f"❌ Error:\n{exec_result['error']}"

                increment_stat(user_id, "messages")
                increment_mode_usage(user_id, current_mode)
                add_exchange(user_id, user_message, resp)
                return resp, False, None, resp

        if tool_need["type"] == "wikipedia":
            search_result = search_wikipedia(tool_need["query"], user_id)
            if search_result["success"]:
                resp = (
                    f"📚 {search_result['title']}\n\n"
                    f"{search_result['summary']}\n\n"
                    f"🔗 {search_result['url']}"
                )
                increment_stat(user_id, "messages")
                increment_mode_usage(user_id, current_mode)
                add_exchange(user_id, user_message, resp)
                return resp, False, None, resp

        if tool_need["type"] == "web":
            search_result = search_web(tool_need["query"], user_id)
            if search_result["success"]:
                resp = (
                    f"🔍 {search_result['title']}\n\n"
                    f"{search_result['content']}\n\n"
                    f"📎 Source: {search_result['source']}"
                )
                increment_stat(user_id, "messages")
                increment_mode_usage(user_id, current_mode)
                add_exchange(user_id, user_message, resp)
                return resp, False, None, resp

        # 4. AI Response - normal (fresh) answer
        system_prompt = PERSONALITY_MODES[current_mode]["prompt"]
        system_prompt += build_feedback_prompt(user_id)

        if detect_ambiguous_question(user_message, history_rows):
            system_prompt += CLARIFYING_ADDITION

        conversation = ""
        if history_rows:
            for row in history_rows[-2:]:  # Only last 2 for clarity
                conversation += f"User: {row['user_msg']}\nAssistant: {row['bot_msg']}\n\n"

        conversation += f"User: {user_message}\nAssistant:"
        prompt = f"{system_prompt}\n\n{conversation}"

    # ========== 2) CONTINUATION (Continue button) ==========
    else:
        previous_answer = user_message or ""

        system_prompt = PERSONALITY_MODES[current_mode]["prompt"]
        system_prompt += build_feedback_prompt(user_id)

        conversation = ""
        if history_rows:
            for row in history_rows[-2:]:
                conversation += f"User: {row['user_msg']}\nAssistant: {row['bot_msg']}\n\n"

        prompt = (
            f"{system_prompt}\n\n"
            f"{conversation}"
            f"Here is your previous partial answer:\n"
            f"{previous_answer}\n\n"
        )

        if is_code:
            prompt += (
                "Continue the code from where it stopped. "
                "Do NOT restart from the beginning and do NOT switch to explanation "
                "until the code is complete.\n"
                "Assistant:"
            )
        else:
            prompt += (
                "Continue the answer from where it stopped. "
                "Do not repeat content you've already written.\n"
                "Assistant:"
            )

    # ========== GENERATION (shared) ==========
    formatted_prompt = f"[INST] {prompt} [/INST]"
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(model.device)

    # Use more tokens for code answers
    max_tokens = CODE_MAX_RESPONSE_TOKENS if is_code else MAX_RESPONSE_TOKENS

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=1,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "[/INST]" in full_response:
        response = full_response.split("[/INST]")[-1].strip()
    else:
        response = full_response.strip()

    # Clean artifacts
    response = clean_response(response)
    response = re.sub(r"^(User:|Assistant:)\s*", "", response, flags=re.MULTILINE)
    response = "\n".join(line for line in response.split("\n") if line.strip())
    response = response.strip()

    # Handle code formatting
    if is_code:
        response = format_code_response(response)

    # Stats
    increment_stat(user_id, "messages")
    increment_mode_usage(user_id, current_mode)

    if is_code:
        increment_stat(user_id, "code_executions")

    # ❗ Do NOT store PDF/OCR tool prompts in conversation history
    if not continue_generation and not (is_pdf_summary or is_ocr_analysis):
        add_exchange(user_id, user_message, response)

    # Heuristic for "is_complete"
    is_complete = len(response) < 1000 or response.endswith((".", "!", "?", "```"))

    return response, not is_complete, prompt, response


async def generate_response_async(user_message, user_id, continue_generation, previous_prompt_cache):
    """Async wrapper"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        partial(generate_response_sync, user_message, user_id, continue_generation, previous_prompt_cache)
    )


# ==================== KEYBOARDS ====================

def mode_keyboard():
    keyboard = [
        [
            InlineKeyboardButton("👨‍🏫 Teacher", callback_data="mode_teacher"),
            InlineKeyboardButton("👋 Friend", callback_data="mode_friend"),
        ],
        [
            InlineKeyboardButton("🔬 Expert", callback_data="mode_expert"),
            InlineKeyboardButton("💻 Coder", callback_data="mode_coder"),
        ],
        [
            InlineKeyboardButton("🔍 Search", callback_data="mode_search")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def reaction_and_continue_keyboard(user_id, show_continue):
    rows = []
    if show_continue:
        rows.append([InlineKeyboardButton("➡️ Continue", callback_data=f"continue_{user_id}")])
    rows.append([
        InlineKeyboardButton("👍", callback_data=f"react_good_{user_id}"),
        InlineKeyboardButton("👎", callback_data=f"react_bad_{user_id}"),
        InlineKeyboardButton("🤔", callback_data=f"react_confused_{user_id}"),
    ])
    return InlineKeyboardMarkup(rows)


# ==================== COMMAND HANDLERS ====================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    clear_history_db(user_id)
    clear_bookmarks_db(user_id)
    clear_stats(user_id)
    clear_reactions(user_id)
    set_user_mode(user_id, "friend")
    response_cache[user_id] = {"prompt": None, "response": None}

    text = (
        "   AI Assistant \n\n"
        "✨ Features:\n"
        "• ⚡ Fast responses\n"
        "• 🧮 Math & derivatives\n"
        "• 📚 Wikipedia search\n"
        "• 💻 Code execution\n"
        "• 📷 OCR for images\n"
        "• 📄 PDF processing\n"
        "• 🎯 5 personality modes\n\n"
        "Commands: /help"
    )
    await update.message.reply_text(text, reply_markup=mode_keyboard())


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "📖 AI Assistant Help\n\n"
        "/start - Reset\n"
        "/help - This message\n"
        "/summary - Short summary & stats\n"
        "/save - Bookmark last answer\n"
        "/bookmarks - View saved answers\n"
        "/stats - Detailed statistics\n"
        "/clear - Clear all your data\n"
        "/teacher /friend /expert /coder /search - Switch modes\n\n"
        "You can send math, questions, code, images, or PDFs!"
    )
    await update.message.reply_text(text)


async def summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    history_rows = get_history(user_id, limit=MAX_HISTORY)
    if not history_rows:
        await update.message.reply_text("No conversation history yet.")
        return

    stats, mode_counts = get_stats(user_id)
    most_used_mode = max(mode_counts, key=mode_counts.get) if mode_counts else get_user_mode(user_id)

    text = (
        f"📊 Summary\n\n"
        f"💬 Messages in memory: {len(history_rows)}\n"
        f"🧮 Calculations: {stats['calculations']}\n"
        f"🔍 Searches: {stats['searches']}\n"
        f"💻 Code runs: {stats['code_executions']}\n"
        f"❤️ Favorite mode: {PERSONALITY_MODES[most_used_mode]['emoji']} {PERSONALITY_MODES[most_used_mode]['name']}"
    )
    await update.message.reply_text(text)


async def bookmarks_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    rows = get_bookmarks_db(user_id)
    if not rows:
        await update.message.reply_text("📚 No bookmarks yet. Use /save to keep answers.")
        return

    lines = ["📚 Bookmarks:\n"]
    for r in rows:
        lines.append(f"{r['id']}. {r['preview']}...")
    await update.message.reply_text("\n\n".join(lines))


async def save_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    cache = response_cache.get(user_id)
    if not cache or not cache.get("response"):
        await update.message.reply_text("❌ No responses to save.")
        return
    ok = add_bookmark(user_id, cache["response"])
    if ok:
        await update.message.reply_text("✓ Bookmarked!")
    else:
        await update.message.reply_text("❌ Bookmark limit reached (20 max).")


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    stats, mode_counts = get_stats(user_id)
    current_mode = get_user_mode(user_id)
    reac = get_reactions(user_id)

    text = (
        f"📊 Statistics\n\n"
        f"🎯 Current mode: {PERSONALITY_MODES[current_mode]['emoji']} {PERSONALITY_MODES[current_mode]['name']}\n"
        f"💬 Messages: {stats['messages']}\n"
        f"🧮 Calculations: {stats['calculations']}\n"
        f"🔍 Searches: {stats['searches']}\n"
        f"💻 Code executions: {stats['code_executions']}\n"
        f"⭐ Bookmarks: {len(get_bookmarks_db(user_id))}/20\n\n"
        f"👍 {reac['good']} | 👎 {reac['bad']} | 🤔 {reac['confused']}"
    )
    await update.message.reply_text(text)


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    clear_history_db(user_id)
    clear_bookmarks_db(user_id)
    clear_stats(user_id)
    clear_reactions(user_id)
    response_cache[user_id] = {"prompt": None, "response": None}
    await update.message.reply_text("✓ All your data has been cleared!")


async def teacher_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    set_user_mode(update.effective_user.id, "teacher")
    await update.message.reply_text("👨‍🏫 Teacher mode activated!")


async def friend_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    set_user_mode(update.effective_user.id, "friend")
    await update.message.reply_text("👋 Friend mode activated!")


async def expert_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    set_user_mode(update.effective_user.id, "expert")
    await update.message.reply_text("🔬 Expert mode activated!")


async def coder_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    set_user_mode(update.effective_user.id, "coder")
    await update.message.reply_text("💻 Coder mode activated!")


async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    set_user_mode(update.effective_user.id, "search")
    await update.message.reply_text("🔍 Search mode activated!")


# ==================== MESSAGE HANDLERS ====================

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    user_id = update.effective_user.id

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    thinking_message = await update.message.reply_text("⏳ Thinking...")

    try:
        resp, need_more, prompt_cache, last_resp = await generate_response_async(
            user_message, user_id, False, None
        )
        response_cache[user_id] = {"prompt": prompt_cache, "response": last_resp or resp}
        kb = reaction_and_continue_keyboard(user_id, need_more)

        try:
            await thinking_message.delete()
        except Exception:
            pass

        try:
            await update.message.reply_text(resp, reply_markup=kb, parse_mode="Markdown")
        except Exception:
            fallback = strip_markdown_for_fallback(resp)
            await update.message.reply_text(fallback, reply_markup=kb)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        await thinking_message.edit_text("❌ Error occurred. Try again.")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    user_id = update.effective_user.id
    doc = message.document
    if not doc:
        return

    f = await doc.get_file()
    os.makedirs("files", exist_ok=True)
    file_path = os.path.join("files", f"{user_id}_{doc.file_unique_id}_{doc.file_name}")
    await f.download_to_drive(custom_path=file_path)
    add_file_record(user_id, "document", file_path, doc.file_name)

    if doc.mime_type == "application/pdf" or doc.file_name.lower().endswith(".pdf"):
        thinking_message = await message.reply_text("📄 Processing PDF...")
        text = extract_text_from_pdf(file_path, user_id)

        if text.startswith("❌"):
            await thinking_message.edit_text(text)
            return

        await thinking_message.edit_text("📄 Analyzing...")

        summary_prompt = (
            "Summarize this PDF content concisely in your own words. "
            "Do not repeat the full text and do not mention these instructions. "
            "Just output a short summary:\n\n"
            f"{text[:3000]}"
        )
        resp, _, _, _ = await generate_response_async(summary_prompt, user_id, False, None)

        try:
            await thinking_message.edit_text(f"📄 **PDF Summary:**\n\n{resp}", parse_mode="Markdown")
        except Exception:
            await thinking_message.edit_text(f"📄 PDF Summary:\n\n{strip_markdown_for_fallback(resp)}")
    else:
        await message.reply_text("📎 Document received!")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    user_id = update.effective_user.id
    photos = message.photo
    if not photos:
        return

    photo = photos[-1]
    f = await photo.get_file()
    os.makedirs("files", exist_ok=True)
    file_path = os.path.join("files", f"{user_id}_{photo.file_unique_id}.jpg")
    await f.download_to_drive(custom_path=file_path)
    add_file_record(user_id, "photo", file_path, "")

    thinking_message = await message.reply_text("📷 Processing...")
    text = extract_text_from_image(file_path, user_id)

    # Error / special cases
    if text.startswith("❌ OCR not available") or text.startswith("❌ Tesseract is not installed") \
            or text.startswith("❌ Error processing image:"):
        await thinking_message.edit_text(text)
        return

    if text.startswith("❌ No readable text found"):
        await thinking_message.edit_text(
            "📷 I couldn't detect any readable text in this image, but I saved the file."
        )
        return

    # Happy path: OCR text found
    await thinking_message.edit_text("📷 Analyzing text...")

    analysis_prompt = (
        "Analyze this text extracted from an image. "
        "Explain briefly what it is about and any key points, "
        "but do not repeat the entire text verbatim:\n\n"
        f"{text[:1500]}"
    )
    resp, _, _, _ = await generate_response_async(analysis_prompt, user_id, False, None)

    try:
        await thinking_message.edit_text(f"📷 **Text Found / Analysis:**\n\n{resp}", parse_mode="Markdown")
    except Exception:
        await thinking_message.edit_text(f"📷 Text Found / Analysis:\n\n{strip_markdown_for_fallback(resp)}")


# ==================== CALLBACKS ====================

async def mode_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    data = query.data

    mode_map = {
        "mode_teacher": "teacher",
        "mode_friend": "friend",
        "mode_expert": "expert",
        "mode_coder": "coder",
        "mode_search": "search",
    }

    mode = mode_map.get(data)
    if mode:
        set_user_mode(user_id, mode)
        info = PERSONALITY_MODES[mode]
        await query.answer()
        await query.edit_message_text(
            f"{info['emoji']} {info['name']} activated!",
            reply_markup=mode_keyboard()
        )


async def continue_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer("⏳ Continuing...")

    cache = response_cache.get(user_id)
    if not cache or not cache.get("prompt") or not cache.get("response"):
        await query.message.reply_text("❌ No previous answer.")
        return

    previous_answer = cache["response"]

    resp, need_more, prompt_cache, last_resp = await generate_response_async(
        previous_answer,  # pass previous partial answer so model continues it
        user_id,
        True,            # continue_generation=True
        cache["prompt"],  # previous prompt (currently unused but kept for future)
    )
    response_cache[user_id] = {"prompt": prompt_cache, "response": last_resp or resp}
    kb = reaction_and_continue_keyboard(user_id, need_more)

    try:
        await query.message.reply_text(resp, reply_markup=kb, parse_mode="Markdown")
    except Exception:
        await query.message.reply_text(strip_markdown_for_fallback(resp), reply_markup=kb)


async def reaction_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    data = query.data

    if data.startswith("react_good_"):
        increment_reaction(user_id, "good")
        await query.answer("👍 Thanks!")
    elif data.startswith("react_bad_"):
        increment_reaction(user_id, "bad")
        await query.answer("👎 I'll improve!")
    elif data.startswith("react_confused_"):
        increment_reaction(user_id, "confused")
        await query.answer("🤔 I'll try to be clearer next time!")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Error: %s", context.error, exc_info=context.error)


# ==================== MAIN ====================

def main():
    print("=" * 60)
    print("🚀 AI ASSISTANT v14 - FIXED EDITION")
    print("=" * 60)
    print("\n✨ Fixed Issues:")
    print("✓ Cleaner code mode & proper ```python``` blocks")
    print("✓ Continue button actually extends previous answers")
    print("✓ Mode-aware Wikipedia & web search")
    print("✓ Better PDF & OCR handling (no history mixing)")
    print("✓ Stronger system prompt leakage cleanup")
    print("✓ Database persistence")
    print("\n" + "=" * 60)

    init_db()
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Commands
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("summary", summary_command))
    application.add_handler(CommandHandler("bookmarks", bookmarks_command))
    application.add_handler(CommandHandler("save", save_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(CommandHandler("teacher", teacher_command))
    application.add_handler(CommandHandler("friend", friend_command))
    application.add_handler(CommandHandler("expert", expert_command))
    application.add_handler(CommandHandler("coder", coder_command))
    application.add_handler(CommandHandler("search", search_command))

    # Callbacks
    application.add_handler(CallbackQueryHandler(mode_callback, pattern="^mode_"))
    application.add_handler(CallbackQueryHandler(continue_callback, pattern="^continue_"))
    application.add_handler(CallbackQueryHandler(reaction_callback, pattern="^react_"))

    # Messages
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

    application.add_error_handler(error_handler)

    print("\n✅ Bot running!")
    print("💬 Ready for messages...\n")

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()