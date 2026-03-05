import streamlit as st
import google.generativeai as genai
from PIL import Image
import edge_tts
import asyncio
from io import BytesIO
from supabase import create_client, Client
import uuid
import time
import tempfile
import ast
import re
import logging
from datetime import datetime

# === CONSTANTE PENTRU LIMITE (FIX MEMORY LEAK) ===
MAX_MESSAGES_IN_MEMORY = 100
MAX_MESSAGES_TO_SEND_TO_AI = 20
MAX_MESSAGES_IN_DB_PER_SESSION = 500
CLEANUP_DAYS_OLD = 7

# === MODELE AI DISPONIBILE ===
AVAILABLE_MODELS = [
    "models/gemini-2.5-flash",
    "models/gemini-1.5-flash", 
    "models/gemini-1.5-pro"
]

# === CONFIGURARE LOGGING ===
def setup_logging():
    """Configurează sistemul de logging."""
    logging.basicConfig(
        filename='profesor_ai.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )

setup_logging()

def log_error(error_type: str, error_msg: str, session_id: str = None):
    """Loghează erori pentru debugging."""
    timestamp = datetime.now().isoformat()
    session_info = f" | Session: {session_id}" if session_id else ""
    log_entry = f"[{timestamp}] {error_type}: {error_msg}{session_info}"
    
    # Log în fișier
    logging.error(log_entry)
    
    # Poți adăuga și notificări pentru tine (email, Telegram etc.) aici
    # Exemplu: trimite_email_alert(log_entry) dacă e eroare critică

# === ISTORIC CONVERSAȚII ===
def invalidate_session_cache():
    """Invalidează cache-ul sesiunilor."""
    st.session_state["_sess_list_ts"] = 0

def get_session_list(limit: int = 20) -> list[dict]:
    """Returnează lista sesiunilor — 2 query-uri totale în loc de N*2."""
    # Folosește cache de 30s pentru a nu re-interoga la fiecare rerun minor
    cache_ts  = st.session_state.get("_sess_list_ts", 0)
    cache_val = st.session_state.get("_sess_list_cache", None)
    if cache_val is not None and (time.time() - cache_ts) < 30:
        return cache_val

    try:
        supabase = get_supabase_client()
        if not supabase:
            return cache_val or []

        # Query 1: sesiunile
        resp = (
            supabase.table("sessions")
            .select("session_id, last_active")
            .order("last_active", desc=True)
            .limit(limit)
            .execute()
        )
        sessions = resp.data or []
        if not sessions:
            return []

        session_ids = [s["session_id"] for s in sessions]

        # Query 2: primul mesaj user + count per sesiune (un singur query)
        hist_resp = (
            supabase.table("history")
            .select("session_id, role, content, timestamp")
            .in_("session_id", session_ids)
            .eq("role", "user")
            .order("timestamp", desc=False)
            .execute()
        )
        hist_rows = hist_resp.data or []

        # Agregare în Python — fără query suplimentare
        first_msg: dict[str, str] = {}
        msg_count: dict[str, int] = {}
        for row in hist_rows:
            sid = row["session_id"]
            msg_count[sid] = msg_count.get(sid, 0) + 1
            if sid not in first_msg:
                txt = row["content"][:60]
                first_msg[sid] = txt + ("..." if len(row["content"]) > 60 else "")

        result = []
        for s in sessions:
            sid = s["session_id"]
            cnt = msg_count.get(sid, 0)
            if cnt > 0:
                result.append({
                    "session_id": sid,
                    "last_active": s["last_active"],
                    "preview": first_msg.get(sid, "Conversație nouă"),
                    "msg_count": cnt,
                })

        st.session_state["_sess_list_cache"] = result
        st.session_state["_sess_list_ts"]    = time.time()
        return result

    except Exception as e:
        error_msg = f"Eroare la încărcarea sesiunilor: {e}"
        log_error("SessionList", error_msg, st.session_state.get("session_id"))
        return cache_val or []


def switch_session(new_session_id: str):
    """Comută la o altă sesiune."""
    st.session_state.session_id = new_session_id
    st.session_state.messages = []
    st.query_params["sid"] = new_session_id
    inject_session_js()
    invalidate_session_cache()  # Invalidează cache-ul


def format_time_ago(timestamp: float) -> str:
    """Formatează timestamp ca timp relativ (ex: '2 ore în urmă')."""
    diff = time.time() - timestamp
    if diff < 60:
        return "acum"
    elif diff < 3600:
        mins = int(diff / 60)
        return f"{mins} min în urmă"
    elif diff < 86400:
        hours = int(diff / 3600)
        return f"{hours}h în urmă"
    else:
        days = int(diff / 86400)
        return f"{days} zile în urmă"


# === SUPABASE CLIENT + FALLBACK ===
@st.cache_resource
def get_supabase_client() -> Client | None:
    """Returnează clientul Supabase (conexiunea e lazy, fără query de test)."""
    try:
        url = st.secrets.get("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_KEY", "")
        if not url or not key:
            return None
        return create_client(url, key)
    except Exception as e:
        log_error("SupabaseClient", str(e))
        return None


def is_supabase_available() -> bool:
    """Returnează statusul Supabase din cache — nu face request la fiecare apel.
    Statusul se actualizează doar când o operație reală eșuează sau reușește."""
    return st.session_state.get("_sb_online", True)


def _mark_supabase_offline():
    """Marchează Supabase ca offline și notifică utilizatorul."""
    was_online = st.session_state.get("_sb_online", True)
    st.session_state["_sb_online"] = False
    if was_online:
        st.toast("⚠️ Baza de date offline — modul local activat.", icon="📴")
        log_error("Supabase", "Conexiune pierdută", st.session_state.get("session_id"))


def _mark_supabase_online():
    """Marchează Supabase ca online și golește coada offline."""
    was_offline = not st.session_state.get("_sb_online", True)
    st.session_state["_sb_online"] = True
    if was_offline:
        st.toast("✅ Conexiunea restabilită!", icon="🟢")
        log_error("Supabase", "Conexiune restabilită", st.session_state.get("session_id"))
        _flush_offline_queue()


# --- Coadă offline: mesaje salvate local când Supabase e down ---
def _get_offline_queue() -> list:
    return st.session_state.setdefault("_offline_queue", [])


def _flush_offline_queue():
    """Trimite mesajele din coada offline la Supabase când revine online."""
    queue = _get_offline_queue()
    if not queue:
        return
    client = get_supabase_client()
    if not client:
        return
    failed = []
    for item in queue:
        try:
            client.table("history").insert(item).execute()
        except Exception as e:
            log_error("OfflineQueue", f"Eroare sincronizare: {e}", item.get("session_id"))
            failed.append(item)
    st.session_state["_offline_queue"] = failed
    if not failed:
        st.toast(f"✅ {len(queue)} mesaje sincronizate cu baza de date.", icon="☁️")

# === VOCI EDGE TTS (VOCE BĂRBAT) ===
VOICE_MALE_RO = "ro-RO-EmilNeural"
VOICE_FEMALE_RO = "ro-RO-AlinaNeural"

# === RATE LIMITING PENTRU TTS ===
def can_use_tts() -> bool:
    """Verifică dacă putem folosi TTS (rate limiting)."""
    now = time.time()
    
    # Resetare zilnică
    if "tts_last_used" not in st.session_state:
        st.session_state.tts_last_used = 0
        st.session_state.tts_count_today = 0
    
    if now - st.session_state.tts_last_used > 86400:  # 24 ore
        st.session_state.tts_count_today = 0
    
    # Limitează la 50 de generări pe zi
    if st.session_state.tts_count_today >= 50:
        return False
    
    st.session_state.tts_last_used = now
    st.session_state.tts_count_today += 1
    return True


st.set_page_config(page_title="Profesor Liceu", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

# Aplică tema dark/light imediat la fiecare rerun
if st.session_state.get("dark_mode", False):
    st.markdown("""
    <script>
    (function() {
        function applyDark() {
            const root = window.parent.document.documentElement;
            root.setAttribute('data-theme', 'dark');
            // Streamlit's internal theme toggle
            const btn = window.parent.document.querySelector('[data-testid="baseButton-headerNoPadding"]');
        }
        applyDark();
        // Re-apply after Streamlit re-renders
        setTimeout(applyDark, 100);
        setTimeout(applyDark, 500);
    })();
    </script>
    <style>
        /* Manual dark mode overrides pentru elementele principale */
        :root { color-scheme: dark; }
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #0e1117 !important;
            color: #fafafa !important;
        }
        [data-testid="stSidebar"] {
            background-color: #161b22 !important;
        }
        .stChatMessage {
            background-color: #1a1f2e !important;
        }
        .stTextArea textarea, .stTextInput input {
            background-color: #1a1f2e !important;
            color: #fafafa !important;
            border-color: #444 !important;
        }
        .stSelectbox > div, .stRadio > div {
            background-color: #1a1f2e !important;
            color: #fafafa !important;
        }
        p, h1, h2, h3, h4, h5, h6, li, label, span {
            color: #fafafa !important;
        }
        .stButton > button {
            border-color: #555 !important;
        }
        hr { border-color: #333 !important; }
        .stExpander { border-color: #333 !important; }
        [data-testid="stChatInput"] {
            background-color: #1a1f2e !important;
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
    .stChatMessage { font-size: 16px; }
    footer { visibility: hidden; }

    /* SVG container - light mode */
    .svg-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        text-align: center;
        margin: 15px 0;
        overflow: auto;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        max-width: 100%;
    }
    .svg-container svg { max-width: 100%; height: auto; }

    /* Dark mode */
    [data-theme="dark"] .svg-container {
        background-color: #1e1e2e;
        border-color: #444;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }

    /* Typing indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 10px 4px;
        font-size: 14px;
        color: #888;
    }
    .typing-dots {
        display: flex;
        gap: 4px;
    }
    .typing-dots span {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #888;
        animation: typing-bounce 1.2s infinite ease-in-out;
    }
    .typing-dots span:nth-child(1) { animation-delay: 0s; }
    .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes typing-bounce {
        0%, 80%, 100% { transform: scale(0.7); opacity: 0.4; }
        40%            { transform: scale(1.0); opacity: 1.0; }
    }
</style>
""", unsafe_allow_html=True)


# === DATABASE FUNCTIONS (SUPABASE) ===
def init_db():
    """Verifică conexiunea la Supabase. Dacă e offline, activează modul local."""
    online = is_supabase_available()
    if not online:
        st.warning("📴 **Modul offline activ** — conversația se păstrează în memorie. "
                   "Istoricul va fi sincronizat automat când conexiunea revine.", icon="⚠️")


def cleanup_old_sessions(days_old: int = CLEANUP_DAYS_OLD):
    """Șterge sesiunile vechi — rulează cel mult o dată pe zi."""
    if time.time() - st.session_state.get("_last_cleanup", 0) < 86400:
        return
    st.session_state["_last_cleanup"] = time.time()
    try:
        supabase = get_supabase_client()
        if not supabase:
            return
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        supabase.table("history").delete().lt("timestamp", cutoff_time).execute()
        supabase.table("sessions").delete().lt("last_active", cutoff_time).execute()
        invalidate_session_cache()  # Invalidează cache-ul după cleanup
    except Exception as e:
        log_error("Cleanup", str(e))


def save_message_to_db(session_id, role, content):
    """Salvează un mesaj în Supabase. Dacă e offline, pune în coada locală."""
    record = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": time.time()
    }
    if not is_supabase_available():
        _get_offline_queue().append(record)
        return
    try:
        client = get_supabase_client()
        if not client:
            _get_offline_queue().append(record)
            return
        client.table("history").insert(record).execute()
        _mark_supabase_online()
        invalidate_session_cache()  # Invalidează cache-ul
    except Exception as e:
        log_error("DBSave", str(e), session_id)
        _mark_supabase_offline()
        _get_offline_queue().append(record)


def load_history_from_db(session_id, limit: int = MAX_MESSAGES_IN_MEMORY):
    """Încarcă istoricul din Supabase. Fallback: returnează ce e deja în session_state."""
    if not is_supabase_available():
        # Offline: întoarce mesajele deja în memorie (dacă există)
        return st.session_state.get("messages", [])[-limit:]
    try:
        client = get_supabase_client()
        if not client:
            return st.session_state.get("messages", [])[-limit:]
        response = (
            client.table("history")
            .select("role, content, timestamp")
            .eq("session_id", session_id)
            .order("timestamp", desc=False)
            .limit(limit)
            .execute()
        )
        return [{"role": row["role"], "content": row["content"]} for row in response.data]
    except Exception as e:
        log_error("DBLoad", str(e), session_id)
        return st.session_state.get("messages", [])[-limit:]


def clear_history_db(session_id):
    """Șterge istoricul pentru o sesiune din Supabase."""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return
        supabase.table("history").delete().eq("session_id", session_id).execute()
        invalidate_session_cache()  # Invalidează cache-ul
    except Exception as e:
        log_error("DBClear", str(e), session_id)


def trim_db_messages(session_id: str):
    """Limitează mesajele din DB pentru o sesiune (FIX MEMORY LEAK)."""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return

        # Numără mesajele sesiunii
        count_resp = (
            supabase.table("history")
            .select("id", count="exact")
            .eq("session_id", session_id)
            .execute()
        )
        count = count_resp.count or 0

        if count > MAX_MESSAGES_IN_DB_PER_SESSION:
            to_delete = count - MAX_MESSAGES_IN_DB_PER_SESSION
            # Obține ID-urile celor mai vechi mesaje
            old_resp = (
                supabase.table("history")
                .select("id")
                .eq("session_id", session_id)
                .order("timestamp", desc=False)
                .limit(to_delete)
                .execute()
            )
            ids_to_delete = [row["id"] for row in old_resp.data]
            if ids_to_delete:
                supabase.table("history").delete().in_("id", ids_to_delete).execute()
    except Exception as e:
        log_error("DBTrim", str(e), session_id)


# === SESSION MANAGEMENT (SUPABASE) ===
def generate_unique_session_id() -> str:
    """Generează un session ID garantat unic."""
    uuid_part = uuid.uuid4().hex[:16]
    time_part = hex(int(time.time() * 1000000))[2:][-8:]
    random_part = uuid.uuid4().hex[:8]
    return f"{uuid_part}{time_part}{random_part}"


def validate_session_id(session_id: str) -> bool:
    """Validează formatul session_id."""
    if not session_id or len(session_id) < 16:
        return False
    # Verifică caractere valide (hex)
    return bool(re.match(r'^[a-f0-9]{32,}$', session_id))


def session_exists_in_db(session_id: str) -> bool:
    """Verifică dacă un session_id există deja în Supabase."""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return False
        response = (
            supabase.table("sessions")
            .select("session_id")
            .eq("session_id", session_id)
            .limit(1)
            .execute()
        )
        return len(response.data) > 0
    except Exception as e:
        log_error("SessionExists", str(e), session_id)
        return False


def register_session(session_id: str):
    """Înregistrează o sesiune nouă în Supabase. Silent dacă offline."""
    if not is_supabase_available():
        return
    try:
        client = get_supabase_client()
        if not client:
            return
        now = time.time()
        client.table("sessions").upsert({
            "session_id": session_id,
            "created_at": now,
            "last_active": now
        }).execute()
        invalidate_session_cache()  # Invalidează cache-ul
    except Exception as e:
        log_error("RegisterSession", str(e), session_id)


def update_session_activity(session_id: str):
    """Actualizează timestamp-ul activității — cel mult o dată la 5 minute."""
    last = st.session_state.get("_last_activity_update", 0)
    if time.time() - last < 300:
        return
    st.session_state["_last_activity_update"] = time.time()
    if not is_supabase_available():
        return
    try:
        client = get_supabase_client()
        if not client:
            return
        client.table("sessions").update({
            "last_active": time.time()
        }).eq("session_id", session_id).execute()
    except Exception as e:
        log_error("UpdateSession", str(e), session_id)


def inject_session_js():
    """
    Injectează JS care sincronizează session_id cu localStorage.
    - La primul load după restart: JS pune sid-ul din localStorage în ?sid= URL.
    - După ce Streamlit setează session_id, JS îl scrie înapoi în localStorage.
    """
    import streamlit.components.v1 as components
    components.html("""
    <script>
    (function() {
        const STORAGE_KEY = 'profesor_session_id';
        const params = new URLSearchParams(window.parent.location.search);
        const sidFromUrl = params.get('sid');

        // Dacă Streamlit a confirmat sid-ul în URL, salvează în localStorage și curăță URL-ul
        if (sidFromUrl && sidFromUrl.length >= 16) {
            localStorage.setItem(STORAGE_KEY, sidFromUrl);
            params.delete('sid');
            const newUrl = window.parent.location.pathname +
                (params.toString() ? '?' + params.toString() : '');
            window.parent.history.replaceState({}, '', newUrl);
            return;
        }

        // Dacă avem session_id în localStorage, pune-l în URL pentru ca Streamlit să îl citească
        const storedId = localStorage.getItem(STORAGE_KEY);
        if (storedId && storedId.length >= 16) {
            params.set('sid', storedId);
            window.parent.location.search = params.toString();
        }
    })();
    </script>
    """, height=0)


def get_or_create_session_id() -> str:
    """
    Obține session ID din: session_state → ?sid= (restaurat din localStorage) → sesiune nouă.
    """
    # 1. Deja în sesiunea curentă Streamlit
    if "session_id" in st.session_state:
        existing_id = st.session_state.session_id
        if existing_id and validate_session_id(existing_id):
            return existing_id

    # 2. Restaurat din localStorage via ?sid= în URL (primul load după restart telefon)
    if "sid" in st.query_params:
        sid_from_storage = st.query_params["sid"]
        if sid_from_storage and validate_session_id(sid_from_storage):
            if session_exists_in_db(sid_from_storage):
                return sid_from_storage

    # 3. Creează sesiune nouă
    for _ in range(10):
        new_id = generate_unique_session_id()
        if not session_exists_in_db(new_id):
            register_session(new_id)
            return new_id

    fallback_id = f"{uuid.uuid4().hex}{int(time.time())}"
    register_session(fallback_id)
    return fallback_id


# === MEMORY MANAGEMENT (FIX MEMORY LEAK) ===
def trim_session_messages():
    """Limitează mesajele din session_state pentru a preveni memory leak."""
    if "messages" in st.session_state:
        current_count = len(st.session_state.messages)
        
        if current_count > MAX_MESSAGES_IN_MEMORY:
            excess = current_count - MAX_MESSAGES_IN_MEMORY
            st.session_state.messages = st.session_state.messages[excess:]
            st.toast(f"📝 Am arhivat {excess} mesaje vechi pentru performanță.", icon="📦")


def get_context_for_ai(messages: list) -> list:
    """Pregătește contextul pentru AI cu limită de mesaje."""
    if len(messages) <= MAX_MESSAGES_TO_SEND_TO_AI:
        return messages[:-1]
    
    first_message = messages[0] if messages else None
    recent_messages = messages[-(MAX_MESSAGES_TO_SEND_TO_AI - 1):-1]
    
    if first_message and first_message not in recent_messages:
        return [first_message] + recent_messages
    return recent_messages


def save_message_with_limits(session_id: str, role: str, content: str):
    """Salvează mesaj și verifică limitele."""
    save_message_to_db(session_id, role, content)
    
    if len(st.session_state.get("messages", [])) % 10 == 0:
        trim_db_messages(session_id)
    
    trim_session_messages()


# === FUNCȚIE PENTRU SELECTAREA MODELULUI AI ===
def get_ai_model():
    """Returnează modelul AI disponibil, cu fallback."""
    current_model = st.session_state.get("ai_model", AVAILABLE_MODELS[0])
    
    # Verifică dacă modelul curent e disponibil
    try:
        genai.get_model(current_model)
        return current_model
    except Exception as e:
        log_error("ModelCheck", f"Model {current_model} indisponibil: {e}")
        # Încearcă următorul model
        for model in AVAILABLE_MODELS:
            try:
                genai.get_model(model)
                st.session_state.ai_model = model
                return model
            except Exception as e2:
                log_error("ModelCheck", f"Fallback {model} indisponibil: {e2}")
                continue
    
    # Fallback la primul model și sperăm
    log_error("ModelCheck", "Niciun model disponibil, folosesc fallback")
    return AVAILABLE_MODELS[0]


# === AUDIO / TTS FUNCTIONS ===

# --- Tabele de date pentru clean_text_for_audio ---

# Unități: (sufix, pronunție) — ordonate de la lung la scurt pentru a evita match greșit
_UNITS: list[tuple[str, str]] = [
    # Rezistență
    ("GΩ", "gigaohmi"), ("MΩ", "megaohmi"), ("kΩ", "kiloohmi"),
    ("mΩ", "miliohmi"), ("μΩ", "microohmi"), ("nΩ", "nanoohmi"), ("Ω", "ohmi"),
    # Temperatură
    ("°C", "grade Celsius"), ("°F", "grade Fahrenheit"), ("°K", "Kelvin"), ("K", "Kelvin"), ("°", "grade"),
    # Tensiune
    ("MV", "megavolți"), ("kV", "kilovolți"), ("mV", "milivolți"), ("μV", "microvolți"), ("V", "volți"),
    # Curent
    ("kA", "kiloamperi"), ("mA", "miliamperi"), ("μA", "microamperi"), ("nA", "nanoamperi"), ("A", "amperi"),
    # Putere
    ("GW", "gigawați"), ("MW", "megawați"), ("kW", "kilowați"), ("mW", "miliwați"), ("μW", "microwați"), ("W", "wați"),
    # Frecvență
    ("THz", "terahertzi"), ("GHz", "gigahertzi"), ("MHz", "megahertzi"), ("kHz", "kilohertzi"), ("mHz", "milihertzi"), ("Hz", "hertzi"),
    # Capacitate
    ("mF", "milifarazi"), ("μF", "microfarazi"), ("nF", "nanofarazi"), ("pF", "picofarazi"), ("F", "farazi"),
    # Inductanță
    ("mH", "milihenry"), ("μH", "microhenry"), ("nH", "nanohenry"), ("H", "henry"),
    # Sarcină electrică
    ("mC", "milicoulombi"), ("μC", "microcoulombi"), ("nC", "nanocoulombi"), ("C", "coulombi"),
    # Câmp magnetic
    ("Wb", "weberi"), ("mT", "militesla"), ("μT", "microtesla"), ("T", "tesla"),
    # Forță
    ("MN", "meganewtoni"), ("kN", "kilonewtoni"), ("mN", "milinewtoni"), ("N", "newtoni"),
    # Energie
    ("kWh", "kilowatt oră"), ("Wh", "watt oră"),
    ("GeV", "gigaelectronvolți"), ("MeV", "megaelectronvolți"), ("keV", "kiloelectronvolți"), ("eV", "electronvolți"),
    ("kcal", "kilocalorii"), ("cal", "calorii"),
    ("GJ", "gigajouli"), ("MJ", "megajouli"), ("kJ", "kilojouli"), ("mJ", "milijouli"), ("J", "jouli"),
    # Presiune
    ("GPa", "gigapascali"), ("MPa", "megapascali"), ("kPa", "kilopascali"), ("hPa", "hectopascali"), ("Pa", "pascali"),
    ("mmHg", "milimetri coloană de mercur"), ("atm", "atmosfere"), ("bar", "bari"),
    # Lungime
    ("km", "kilometri"), ("dm", "decimetri"), ("cm", "centimetri"), ("mm", "milimetri"),
    ("μm", "micrometri"), ("nm", "nanometri"), ("pm", "picometri"), ("Å", "angstromi"), ("m", "metri"),
    # Masă
    ("kg", "kilograme"), ("mg", "miligrame"), ("μg", "micrograme"), ("ng", "nanograme"), ("g", "grame"), ("t", "tone"),
    # Volum
    ("mL", "mililitri"), ("ml", "mililitri"), ("μL", "microlitri"), ("L", "litri"), ("l", "litri"),
    ("dm³", "decimetri cubi"), ("cm³", "centimetri cubi"), ("mm³", "milimetri cubi"), ("m³", "metri cubi"),
    # Timp
    ("ms", "milisecunde"), ("μs", "microsecunde"), ("ns", "nanosecunde"), ("ps", "picosecunde"),
    ("min", "minute"), ("s", "secunde"), ("h", "ore"),
    # Suprafață
    ("km²", "kilometri pătrați"), ("m²", "metri pătrați"), ("dm²", "decimetri pătrați"),
    ("cm²", "centimetri pătrați"), ("mm²", "milimetri pătrați"), ("ha", "hectare"),
    # Viteză & derivate
    ("m/s²", "metri pe secundă la pătrat"), ("m/s", "metri pe secundă"), ("km/h", "kilometri pe oră"),
    ("km/s", "kilometri pe secundă"), ("cm/s", "centimetri pe secundă"),
    ("rad/s", "radiani pe secundă"), ("rpm", "rotații pe minut"),
    # Densitate, presiune compusă
    ("kg/m³", "kilograme pe metru cub"), ("g/cm³", "grame pe centimetru cub"), ("g/mL", "grame pe mililitru"),
    ("N/m²", "newtoni pe metru pătrat"), ("N/m", "newtoni pe metru"),
    ("J/kg", "jouli pe kilogram"), ("J/mol", "jouli pe mol"),
    ("W/m²", "wați pe metru pătrat"), ("V/m", "volți pe metru"), ("A/m", "amperi pe metru"),
    # Chimie
    ("mol/L", "moli pe litru"), ("mol/l", "moli pe litru"),
    ("g/mol", "grame pe mol"), ("kg/mol", "kilograme pe mol"),
    ("mol", "moli"), ("M", "molar"),
    # Radiație & optică
    ("Bq", "becquereli"), ("Gy", "gray"), ("Sv", "sievert"),
    ("cd", "candele"), ("lm", "lumeni"), ("lx", "lucși"),
    # Unghiuri
    ("rad", "radiani"), ("sr", "steradiani"),
]

# Simboluri și combinații speciale: (literal, înlocuitor)
_SYMBOLS: dict[str, str] = {
    ">=": " mai mare sau egal cu ", "<=": " mai mic sau egal cu ",
    "!=": " diferit de ", "==": " egal cu ", "<>": " diferit de ",
    ">>": " mult mai mare decât ", "<<": " mult mai mic decât ",
    "->": " implică ", "<-": " provine din ", "<->": " echivalent cu ", "=>": " rezultă că ",
    "...": " ", "…": " ", "N·m": " newton metri ", "N*m": " newton metri ", "kW·h": " kilowatt oră ",
    "α": " alfa ", "β": " beta ", "γ": " gama ", "δ": " delta ", "ε": " epsilon ",
    "ζ": " zeta ", "η": " eta ", "θ": " teta ", "ι": " iota ", "κ": " kapa ",
    "λ": " lambda ", "μ": " miu ", "ν": " niu ", "ξ": " csi ", "ο": " omicron ",
    "π": " pi ", "ρ": " ro ", "σ": " sigma ", "ς": " sigma ", "τ": " tau ",
    "υ": " ipsilon ", "φ": " fi ", "χ": " hi ", "ψ": " psi ", "ω": " omega ",
    "Α": " alfa ", "Β": " beta ", "Γ": " gama ", "Δ": " delta ", "Ε": " epsilon ",
    "Ζ": " zeta ", "Η": " eta ", "Θ": " teta ", "Ι": " iota ", "Κ": " kapa ",
    "Λ": " lambda ", "Μ": " miu ", "Ν": " niu ", "Ξ": " csi ", "Ο": " omicron ",
    "Π": " pi ", "Ρ": " ro ", "Σ": " sigma ", "Τ": " tau ", "Υ": " ipsilon ",
    "Φ": " fi ", "Χ": " hi ", "Ψ": " psi ", "Ω": " omega ",
    "∞": " infinit ", "∑": " suma ", "∏": " produsul ", "∫": " integrala ",
    "∂": " derivata parțială ", "√": " radical din ", "∛": " radical de ordin 3 din ",
    "∜": " radical de ordin 4 din ", "±": " plus minus ", "∓": " minus plus ",
    "×": " ori ", "÷": " împărțit la ", "≠": " diferit de ", "≈": " aproximativ egal cu ",
    "≡": " identic cu ", "≤": " mai mic sau egal cu ", "≥": " mai mare sau egal cu ",
    "≪": " mult mai mic decât ", "≫": " mult mai mare decât ", "∝": " proporțional cu ",
    "∈": " aparține lui ", "∉": " nu aparține lui ", "⊂": " inclus în ", "⊃": " include ",
    "⊆": " inclus sau egal cu ", "⊇": " include sau egal cu ",
    "∪": " reunit cu ", "∩": " intersectat cu ", "∅": " mulțimea vidă ",
    "∀": " pentru orice ", "∃": " există ", "∄": " nu există ",
    "∴": " deci ", "∵": " deoarece ",
    "→": " implică ", "←": " rezultă din ", "↔": " echivalent cu ",
    "⇒": " rezultă că ", "⇐": " provine din ", "⇔": " dacă și numai dacă ",
    "↑": " crește ", "↓": " scade ", "°": " grade ", "′": " ", "″": " ",
    "‰": " la mie ", "∠": " unghiul ", "⊥": " perpendicular pe ", "∥": " paralel cu ",
    "△": " triunghiul ", "□": " ", "○": " ", "★": " ", "☆": " ",
    "✓": " corect ", "✗": " greșit ", "✘": " greșit ",
    ">": " mai mare decât ", "<": " mai mic decât ", "=": " egal ",
    "+": " plus ", "−": " minus ", "—": " ", "–": " ",
    "·": " ori ", "•": " ", "∙": " ori ", "⋅": " ori ",
    "⁰": " la puterea 0 ", "¹": " la puterea 1 ", "²": " la pătrat ", "³": " la cub ",
    "⁴": " la puterea 4 ", "⁵": " la puterea 5 ", "⁶": " la puterea 6 ",
    "⁷": " la puterea 7 ", "⁸": " la puterea 8 ", "⁹": " la puterea 9 ",
    "⁺": " plus ", "⁻": " minus ", "⁼": " egal ",
    "₀": " indice 0 ", "₁": " indice 1 ", "₂": " indice 2 ", "₃": " indice 3 ",
    "₄": " indice 4 ", "₅": " indice 5 ", "₆": " indice 6 ", "₇": " indice 7 ",
    "₈": " indice 8 ", "₉": " indice 9 ", "₊": " plus ", "₋": " minus ", "₌": " egal ",
    "ₐ": " indice a ", "ₑ": " indice e ", "ₕ": " indice h ", "ᵢ": " indice i ",
    "ⱼ": " indice j ", "ₖ": " indice k ", "ₗ": " indice l ", "ₘ": " indice m ",
    "ₙ": " indice n ", "ₒ": " indice o ", "ₚ": " indice p ", "ᵣ": " indice r ",
    "ₛ": " indice s ", "ₜ": " indice t ", "ᵤ": " indice u ", "ᵥ": " indice v ", "ₓ": " indice x ",
    "ᵦ": " indice beta ", "ᵧ": " indice gama ", "ᵨ": " indice ro ", "ᵩ": " indice fi ", "ᵪ": " indice hi ",
    "ᵃ": " la puterea a ", "ᵇ": " la puterea b ", "ᶜ": " la puterea c ", "ᵈ": " la puterea d ",
    "ᵉ": " la puterea e ", "ᶠ": " la puterea f ", "ᵍ": " la puterea g ", "ʰ": " la puterea h ",
    "ⁱ": " la puterea i ", "ʲ": " la puterea j ", "ᵏ": " la puterea k ", "ˡ": " la puterea l ",
    "ᵐ": " la puterea m ", "ⁿ": " la puterea n ", "ᵒ": " la puterea o ", "ᵖ": " la puterea p ",
    "ʳ": " la puterea r ", "ˢ": " la puterea s ", "ᵗ": " la puterea t ", "ᵘ": " la puterea u ",
    "ᵛ": " la puterea v ", "ʷ": " la puterea w ", "ˣ": " la puterea x ", "ʸ": " la puterea y ", "ᶻ": " la puterea z ",
    "½": " o doime ", "⅓": " o treime ", "⅔": " două treimi ", "¼": " un sfert ", "¾": " trei sferturi ",
    "⅕": " o cincime ", "⅖": " două cincimi ", "⅗": " trei cincimi ", "⅘": " patru cincimi ",
    "⅙": " o șesime ", "⅚": " cinci șesimi ", "⅛": " o optime ", "⅜": " trei optimi ",
    "⅝": " cinci optimi ", "⅞": " șapte optimi ",
    "%": " procent ", "&": " și ", "#": " numărul ", "~": " aproximativ ",
    "≅": " congruent cu ", "≃": " aproximativ egal cu ", "|": " ", "‖": " ", "⋯": " ",
    "∧": " și ", "∨": " sau ", "¬": " negația lui ", "∎": " ",
    "ℕ": " mulțimea numerelor naturale ", "ℤ": " mulțimea numerelor întregi ",
    "ℚ": " mulțimea numerelor raționale ", "ℝ": " mulțimea numerelor reale ",
    "ℂ": " mulțimea numerelor complexe ", "℃": " grade Celsius ", "℉": " grade Fahrenheit ",
    "Å": " angstrom ", "№": " numărul ",
}

# Comenzi LaTeX: (pattern, replacement)
_LATEX_PATTERNS: list[tuple[str, str]] = [
    (r'\\sqrt\[(\d+)\]\{([^}]+)\}', r' radical de ordin \1 din \2 '),
    (r'\\sqrt\{([^}]+)\}', r' radical din \1 '),
    (r'\\d?frac\{([^}]+)\}\{([^}]+)\}', r' \1 supra \2 '),
    (r'\^\{([^}]+)\}', r' la puterea \1 '), (r'\^(\d+)', r' la puterea \1 '),
    (r'_\{([^}]+)\}', r' indice \1 '),     (r'_(\d+)', r' indice \1 '),
    (r'\\alpha', ' alfa '), (r'\\beta', ' beta '), (r'\\gamma', ' gama '),
    (r'\\delta', ' delta '), (r'\\(?:var)?epsilon', ' epsilon '),
    (r'\\zeta', ' zeta '), (r'\\eta', ' eta '), (r'\\(?:var)?theta', ' teta '),
    (r'\\iota', ' iota '), (r'\\kappa', ' kapa '), (r'\\lambda', ' lambda '),
    (r'\\mu', ' miu '), (r'\\nu', ' niu '), (r'\\xi', ' csi '),
    (r'\\(?:var)?pi', ' pi '), (r'\\(?:var)?rho', ' ro '),
    (r'\\(?:var)?sigma', ' sigma '), (r'\\tau', ' tau '), (r'\\upsilon', ' ipsilon '),
    (r'\\(?:var)?phi', ' fi '), (r'\\chi', ' hi '), (r'\\psi', ' psi '),
    (r'\\(?:var)?omega', ' omega '),
    (r'\\Gamma', ' gama '), (r'\\Delta', ' delta '), (r'\\Theta', ' teta '),
    (r'\\Lambda', ' lambda '), (r'\\Xi', ' csi '), (r'\\Pi', ' pi '),
    (r'\\Sigma', ' sigma '), (r'\\Upsilon', ' ipsilon '), (r'\\Phi', ' fi '),
    (r'\\Psi', ' psi '), (r'\\Omega', ' omega '),
    (r'\\times', ' ori '), (r'\\cdot', ' ori '), (r'\\div', ' împărțit la '),
    (r'\\pm', ' plus minus '), (r'\\mp', ' minus plus '),
    (r'\\(?:leq?)', ' mai mic sau egal cu '), (r'\\(?:geq?)', ' mai mare sau egal cu '),
    (r'\\(?:neq?)', ' diferit de '), (r'\\approx', ' aproximativ egal cu '),
    (r'\\equiv', ' echivalent cu '), (r'\\sim', ' similar cu '),
    (r'\\propto', ' proporțional cu '), (r'\\infty', ' infinit '),
    (r'\\sum', ' suma '), (r'\\prod', ' produsul '),
    (r'\\iiint', ' integrala triplă '), (r'\\iint', ' integrala dublă '),
    (r'\\oint', ' integrala pe contur '), (r'\\int', ' integrala '),
    (r'\\lim', ' limita '), (r'\\log', ' logaritm de '), (r'\\ln', ' logaritm natural de '),
    (r'\\lg', ' logaritm zecimal de '), (r'\\exp', ' exponențiala de '),
    (r'\\sin', ' sinus de '), (r'\\cos', ' cosinus de '),
    (r'\\(?:tg|tan)', ' tangentă de '), (r'\\(?:ctg|cot)', ' cotangentă de '),
    (r'\\sec', ' secantă de '), (r'\\csc', ' cosecantă de '),
    (r'\\arcsin', ' arc sinus de '), (r'\\arccos', ' arc cosinus de '),
    (r'\\(?:arctg|arctan)', ' arc tangentă de '),
    (r'\\sinh', ' sinus hiperbolic de '), (r'\\cosh', ' cosinus hiperbolic de '),
    (r'\\tanh', ' tangentă hiperbolică de '),
    (r'\\(?:right|left)?arrow', ' implică '), (r'\\to\b', ' tinde la '),
    (r'\\Rightarrow', ' rezultă că '), (r'\\Leftarrow', ' este implicat de '),
    (r'\\[Ll]eftrightarrow', ' echivalent cu '), (r'\\Leftrightarrow', ' dacă și numai dacă '),
    (r'\\forall', ' pentru orice '), (r'\\exists', ' există '), (r'\\nexists', ' nu există '),
    (r'\\in\b', ' aparține lui '), (r'\\notin', ' nu aparține lui '),
    (r'\\subseteq', ' inclus sau egal cu '), (r'\\supseteq', ' include sau egal cu '),
    (r'\\subset', ' inclus în '), (r'\\supset', ' include '),
    (r'\\cup', ' reunit cu '), (r'\\cap', ' intersectat cu '),
    (r'\\(?:empty[Ss]et|varnothing)', ' mulțimea vidă '),
    (r'\\mathbb\{R\}', ' mulțimea numerelor reale '),
    (r'\\mathbb\{N\}', ' mulțimea numerelor naturale '),
    (r'\\mathbb\{Z\}', ' mulțimea numerelor întregi '),
    (r'\\mathbb\{Q\}', ' mulțimea numerelor raționale '),
    (r'\\mathbb\{C\}', ' mulțimea numerelor complexe '),
    (r'\\partial', ' derivata parțială '), (r'\\nabla', ' nabla '),
    (r'\\(?:degree|circ)\b', ' grad '), (r'\\(?:angle|measuredangle)', ' unghiul '),
    (r'\\perp', ' perpendicular pe '), (r'\\parallel', ' paralel cu '),
    (r'\\triangle', ' triunghiul '), (r'\\square', ' pătratul '),
    (r'\\therefore', ' deci '), (r'\\because', ' deoarece '),
    (r'\\lt\b', ' mai mic decât '), (r'\\gt\b', ' mai mare decât '),
]

# Regex precompilat pentru unități (număr + unitate) - cu lookbehind pentru cuvinte
_NUM = r'(\d+[.,]?\d*)'
_UNIT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'(?<!\w)' + _NUM + r'\s*' + re.escape(unit) + r'\b'), r'\1 ' + pron)
    for unit, pron in _UNITS
]


def clean_text_for_audio(text: str) -> str:
    """Curăță textul de LaTeX, SVG, Markdown pentru TTS."""
    if not text:
        return ""

    # 1. Elimină blocuri SVG complet
    text = re.sub(r'\[\[DESEN_SVG\]\].*?\[\[/DESEN_SVG\]\]',
                  ' Am desenat o figură pentru tine. ', text, flags=re.DOTALL)
    text = re.sub(r'<svg.*?</svg>', ' ', text, flags=re.DOTALL)

    # 2. Unități de măsură — aplică din tabela precompilată
    for pattern, replacement in _UNIT_PATTERNS:
        text = pattern.sub(replacement, text)

    # 3. Indici cu underscore (P_r, V_0 etc.)
    text = re.sub(r'([A-Za-zα-ωΑ-Ω])\s*_\s*\{([^}]+)\}', r'\1 indice \2', text)
    text = re.sub(r'([A-Za-zα-ωΑ-Ω])\s*_\s*([A-Za-z0-9α-ωΑ-Ω]+)', r'\1 indice \2', text)

    # 4. Simboluri și combinații speciale — aplică din tabela _SYMBOLS
    for symbol, replacement in _SYMBOLS.items():
        text = text.replace(symbol, replacement)

    # 5. Punctuație matematică
    text = re.sub(r'(\d)\s*:\s*(\d)', r'\1 este la \2', text)
    text = re.sub(r'(\d+)\s*/\s*(\d+)', r'\1 supra \2', text)
    text = re.sub(r':\s*$', '.', text)
    text = re.sub(r':\s*\n', '.\n', text)
    text = re.sub(r'(\w):\s+', r'\1. ', text)

    # 6. LaTeX — aplică din tabela _LATEX_PATTERNS
    for pattern, replacement in _LATEX_PATTERNS:
        text = re.sub(pattern, replacement, text)

    # 7. Elimină delimitatorii LaTeX rămași
    text = re.sub(r'\$\$([^$]+)\$\$', r' \1 ', text)
    text = re.sub(r'\$([^$]+)\$', r' \1 ', text)
    text = re.sub(r'\\\[(.+?)\\\]', r' \1 ', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.+?)\\\)', r' \1 ', text)

    # 8. Curăță comenzile LaTeX rămase
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'[{}\\]', '', text)

    # 9. Elimină Markdown
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # 10. Elimină HTML rămas
    text = re.sub(r'<[^>]+>', '', text)

    # 11. Curăță caractere speciale rămase și spații
    text = re.sub(r'[│▌►◄■▪▫\[\](){}]', ' ', text)
    text = re.sub(r'\s*:\s*', '. ', text)
    text = re.sub(r'\s+', ' ', text)

    # 12. Limitează lungimea
    text = text.strip()
    if len(text) > 3000:
        text = text[:3000]
        last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        if last_period > 2500:
            text = text[:last_period + 1]

    return text


async def _generate_audio_edge_tts(text: str, voice: str = VOICE_MALE_RO) -> bytes:
    """Generează audio folosind Edge TTS (async)."""
    try:
        clean_text = clean_text_for_audio(text)
        
        if not clean_text or len(clean_text.strip()) < 10:
            return None
        
        communicate = edge_tts.Communicate(clean_text, voice)
        audio_data = BytesIO()
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])
        
        audio_data.seek(0)
        return audio_data.getvalue()
        
    except Exception as e:
        log_error("TTS", str(e))
        return None


def generate_professor_voice(text: str, voice: str = VOICE_MALE_RO) -> BytesIO:
    """Wrapper sincron pentru Edge TTS - voce de bărbat (Domnul Profesor)."""
    # Verifică rate limiting
    if not can_use_tts():
        st.caption("🔇 Limită zilnică de voce atinsă (50 generări/zi)")
        return None
        
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            audio_bytes = loop.run_until_complete(_generate_audio_edge_tts(text, voice))
        finally:
            loop.close()
        
        if audio_bytes:
            audio_file = BytesIO(audio_bytes)
            audio_file.seek(0)
            return audio_file
        return None
        
    except Exception as e:
        log_error("Voice", str(e))
        return None


# === SVG FUNCTIONS (FIX SVG FĂRĂ CLOSE TAG) ===
def repair_svg(svg_content: str) -> str:
    """Repară SVG incomplet sau malformat."""
    if not svg_content:
        return None
    
    svg_content = svg_content.strip()
    
    has_svg_open = bool(re.search(r'<svg[^>]*>', svg_content, re.IGNORECASE))
    has_svg_close = '</svg>' in svg_content.lower()
    
    if not has_svg_open:
        svg_content = f'''<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg" 
                             style="max-width: 100%; height: auto; background-color: white;">
            {svg_content}
        </svg>'''
        return svg_content
    
    if has_svg_open and not has_svg_close:
        svg_content = svg_content + '\n</svg>'
    
    if not has_svg_open and has_svg_close:
        svg_content = f'<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">\n{svg_content}'
    
    svg_content = repair_unclosed_tags(svg_content)
    
    if 'xmlns=' not in svg_content:
        svg_content = svg_content.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    
    if 'viewBox=' not in svg_content.lower():
        svg_content = svg_content.replace('<svg', '<svg viewBox="0 0 800 600"', 1)
    
    return svg_content


def repair_unclosed_tags(svg_content: str) -> str:
    """Repară tag-uri SVG comune care nu sunt închise corect."""
    self_closing_tags = ['path', 'rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon', 'image', 'use']
    
    for tag in self_closing_tags:
        pattern = rf'<{tag}([^>]*[^/])>'
        
        def fix_tag(match):
            attrs = match.group(1)
            return f'<{tag}{attrs}/>'
        
        svg_content = re.sub(pattern, fix_tag, svg_content)
    
    text_opens = len(re.findall(r'<text[^>]*>', svg_content))
    text_closes = len(re.findall(r'</text>', svg_content))
    
    if text_opens > text_closes:
        for _ in range(text_opens - text_closes):
            svg_content = svg_content.replace('</svg>', '</text></svg>')
    
    g_opens = len(re.findall(r'<g[^>]*>', svg_content))
    g_closes = len(re.findall(r'</g>', svg_content))
    
    if g_opens > g_closes:
        for _ in range(g_opens - g_closes):
            svg_content = svg_content.replace('</svg>', '</g></svg>')
    
    return svg_content


def validate_svg(svg_content: str) -> tuple:
    """Validează SVG și returnează (is_valid, error_message)."""
    if not svg_content:
        return False, "SVG gol"
    
    if '<svg' not in svg_content.lower():
        return False, "Lipsește tag-ul <svg>"
    
    if '</svg>' not in svg_content.lower():
        return False, "Lipsește tag-ul </svg>"
    
    visual_elements = ['path', 'rect', 'circle', 'ellipse', 'line', 'text', 'polygon', 'polyline', 'image']
    has_content = any(f'<{elem}' in svg_content.lower() for elem in visual_elements)
    
    if not has_content:
        return False, "SVG fără elemente vizuale"
    
    return True, "OK"


def render_message_with_svg(content: str):
    """Renderează mesajul cu suport îmbunătățit pentru SVG."""
    has_svg_markers = '[[DESEN_SVG]]' in content or '<svg' in content.lower()
    has_svg_elements = any(tag in content.lower() for tag in ['<path', '<rect', '<circle', '<line', '<polygon'])
    
    if has_svg_markers or (has_svg_elements and 'stroke=' in content):
        svg_code = None
        before_text = ""
        after_text = ""
        
        if '[[DESEN_SVG]]' in content:
            parts = content.split('[[DESEN_SVG]]')
            before_text = parts[0]
            if len(parts) > 1 and '[[/DESEN_SVG]]' in parts[1]:
                inner_parts = parts[1].split('[[/DESEN_SVG]]')
                svg_code = inner_parts[0]
                after_text = inner_parts[1] if len(inner_parts) > 1 else ""
            elif len(parts) > 1:
                svg_code = parts[1]
        elif '<svg' in content.lower():
            svg_match = re.search(r'<svg.*?</svg>', content, re.DOTALL | re.IGNORECASE)
            if svg_match:
                svg_code = svg_match.group(0)
                before_text = content[:svg_match.start()]
                after_text = content[svg_match.end():]
            else:
                svg_start = content.lower().find('<svg')
                if svg_start != -1:
                    before_text = content[:svg_start]
                    svg_code = content[svg_start:]
        
        if svg_code:
            svg_code = repair_svg(svg_code)
            is_valid, error = validate_svg(svg_code)
            
            if is_valid:
                if before_text.strip():
                    st.markdown(before_text.strip())
                
                st.markdown(
                    f'<div class="svg-container">{svg_code}</div>',
                    unsafe_allow_html=True
                )
                
                if after_text.strip():
                    st.markdown(after_text.strip())
                return
            else:
                st.warning(f"⚠️ Desenul nu a putut fi afișat corect: {error}")
    
    clean_content = content
    clean_content = re.sub(r'\[\[DESEN_SVG\]\]', '\n🎨 *Desen:*\n', clean_content)
    clean_content = re.sub(r'\[\[/DESEN_SVG\]\]', '\n', clean_content)
    
    st.markdown(clean_content)


# === INIȚIALIZARE ===
init_db()
cleanup_old_sessions(CLEANUP_DAYS_OLD)

session_id = get_or_create_session_id()
st.session_state.session_id = session_id
update_session_activity(session_id)

# Sincronizează session_id cu localStorage — doar la primul load
if not st.session_state.get("_js_injected"):
    st.query_params["sid"] = session_id
    inject_session_js()
    st.session_state["_js_injected"] = True


# === API KEYS ===
raw_keys = None
if "GOOGLE_API_KEYS" in st.secrets:
    raw_keys = st.secrets["GOOGLE_API_KEYS"]
elif "GOOGLE_API_KEY" in st.secrets:
    raw_keys = [st.secrets["GOOGLE_API_KEY"]]
else:
    k = st.sidebar.text_input("API Key (Manual):", type="password")
    raw_keys = [k] if k else []

keys = []
if raw_keys:
    if isinstance(raw_keys, str):
        try:
            raw_keys = ast.literal_eval(raw_keys)
        except:
            raw_keys = [raw_keys]
    if isinstance(raw_keys, list):
        for k in raw_keys:
            if k and isinstance(k, str):
                clean_k = k.strip().strip('"').strip("'")
                if clean_k:
                    keys.append(clean_k)

if not keys:
    st.error("❌ Nu am găsit nicio cheie API validă.")
    st.stop()

if "key_index" not in st.session_state:
    st.session_state.key_index = 0


# === MATERII ===
MATERII = {
    "🎓 Toate materiile": None,
    "📐 Matematică":      "matematică",
    "⚡ Fizică":          "fizică",
    "🧪 Chimie":          "chimie",
    "📖 Română":          "limba și literatura română",
    "🇫🇷 Franceză":       "limba franceză",
    "🇬🇧 Engleză":        "limba engleză",
    "🌍 Geografie":       "geografie",
    "🏛️ Istorie":         "istorie",
    "💻 Informatică":     "informatică",
    "🧬 Biologie":        "biologie",
}


def get_system_prompt(materie: str | None = None) -> str:
    """Returnează System Prompt adaptat materiei selectate."""

    if materie:
        rol_line = (
            f"ROL: Ești un profesor de liceu din România specializat în {materie.upper()}, "
            f"bărbat, cu experiență în pregătirea pentru BAC. "
            f"Răspunde EXCLUSIV la întrebări legate de {materie}. "
            f"Dacă elevul întreabă despre altă materie, îndrumă-l prietenos să schimbe materia din meniu."
        )
    else:
        rol_line = (
            "ROL: Ești un profesor de liceu din România, universal "
            "(Mate, Fizică, Chimie, Literatură și Gramatică Română, Franceză, Engleză, "
            "Geografie, Istorie, Informatică, Biologie), bărbat, cu experiență în pregătirea pentru BAC."
        )

    return r"""
ROL: """ + rol_line + r"""

    REGULI DE IDENTITATE (STRICT):
    1. Folosește EXCLUSIV genul masculin când vorbești despre tine.
       - Corect: "Sunt sigur", "Sunt pregătit", "Am fost atent", "Sunt bucuros".
       - GREȘIT: "Sunt sigură", "Sunt pregătită".
    2. Te prezinți ca "Domnul Profesor" sau "Profesorul tău virtual".

    TON ȘI ADRESARE (CRITIC):
    3. Vorbește DIRECT, la persoana I singular.
       - CORECT: "Salut, sunt aici să te ajut." / "Te ascult." / "Sunt pregătit."
       - GREȘIT: "Domnul profesor este aici." / "Profesorul te va ajuta."
    4. Fii cald, natural, apropiat și scurt. Evită introducerile pompoase.
    5. NU SALUTA în fiecare mesaj. Salută DOAR la începutul unei conversații noi.
    6. Dacă elevul pune o întrebare directă, răspunde DIRECT la subiect, fără introduceri de genul "Salut, desigur...".
    7. Folosește "Salut" sau "Te salut" în loc de formule foarte oficiale.

    REGULĂ STRICTĂ: Predă exact ca la școală (nivel Gimnaziu/Liceu).
    NU confunda elevul cu detalii despre "aproximări" sau "lumea reală" (frecare, erori) decât dacă problema o cere specific.

    GHID DE COMPORTAMENT:
    1. MATEMATICĂ:
       - Lucrează cu valori exacte ($\sqrt{2}$, $\pi$) sau standard.
       - Dacă rezultatul e $\sqrt{2}$, lasă-l $\sqrt{2}$. Nu spune "care este aproximativ 1.41".
       - Nu menționa că $\pi$ e infinit; folosește valorile din manual fără comentarii suplimentare.
       - Explică logica din spate, nu doar calculul.
       - Dacă rezultatul e rad(2), lasă-l rad(2). Nu îl calcula aproximativ.
       - Folosește LaTeX ($...$) pentru toate formulele.

    2. FIZICĂ/CHIMIE:
       - Presupune automat "condiții ideale".
       - Tratează problema exact așa cum apare în culegere.
       - Nu menționa frecarea cu aerul, pierderile de căldură sau imperfecțiunile aparatelor de măsură.
       - Tratează problema exact așa cum apare în culegere, într-un univers matematic perfect.

    3. LIMBA ȘI LITERATURA ROMÂNĂ (CRITIC):
       - Respectă STRICT programa școlară de BAC din România și canoanele criticii (G. Călinescu, E. Lovinescu, T. Vianu).
       - ATENȚIE MAJORA: Ion Creangă (Harap-Alb) este Basm Cult, dar specificul lui este REALISMUL (umanizarea fantasticului, oralitatea), nu romantismul.
       - La poezie: Încadrează corect (Romantism - Eminescu, Modernism - Blaga/Arghezi, Simbolism - Bacovia).
       - Structurează răspunsurile ca un eseu de BAC (Ipoteză -> Argumente (pe text) -> Concluzie).

    4. STIL DE PREDARE:
           - Explică simplu, cald și prietenos. Evită "limbajul de lemn".
           - Folosește analogii pentru concepte grele (ex: "Curentul e ca debitul apei").
           - La teorie: Definiție -> Exemplu Concret -> Aplicație.
           - La probleme: Explică pașii logici ("Facem asta pentru că..."), nu da doar calculul.

    5. MATERIALE UPLOADATE (Cărți/PDF):
           - Dacă primești o carte, păstrează sensul original în rezumate/traduceri.
           - Dacă elevul încarcă o poză sau un PDF, analizează tot conținutul înainte de a răspunde.
           - Păstrează sensul original al textelor din manuale.

    6. FUNCȚIE SPECIALĂ - DESENARE (SVG):
        Dacă elevul cere un desen, o diagramă sau o hartă:
        1. Ești OBLIGAT să generezi cod SVG valid.
        2. Codul trebuie încadrat STRICT între tag-uri:
           [[DESEN_SVG]]
           <svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
              <!-- Codul tău aici -->
           </svg>
           [[/DESEN_SVG]]
        3. IMPORTANT: Nu uita tag-ul de deschidere <svg> și cel de închidere </svg>!

        REGULI HĂRȚI (GEOGRAFIE):
        - Nu desena pătrate. Folosește <path> pentru contururi.
        - Râurile = linii albastre.
        - Adaugă etichete text (<text>).
"""


# System prompt inițial (fără materie selectată)
SYSTEM_PROMPT = get_system_prompt()



safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]



# ============================================================
# === SIMULARE BAC ===
# ============================================================

MATERII_BAC = {
    "📐 Matematică": {
        "cod": "matematica",
        "profile": ["M1 - Mate-Info", "M2 - Științe ale naturii"],
        "subiecte": ["Algebră", "Analiză matematică", "Geometrie"],
        "timp_minute": 180,
        "punctaj_total": 100,
    },
    "📖 Română": {
        "cod": "romana",
        "profile": ["Toate profilurile"],
        "subiecte": ["Text literar", "Text nonliterar", "Redactare eseu"],
        "timp_minute": 180,
        "punctaj_total": 100,
    },
    "⚡ Fizică": {
        "cod": "fizica",
        "profile": ["Mate-Info", "Științe ale naturii"],
        "subiecte": ["Mecanică", "Termodinamică", "Electricitate", "Optică"],
        "timp_minute": 180,
        "punctaj_total": 100,
    },
    "🧪 Chimie": {
        "cod": "chimie",
        "profile": ["Chimie anorganică", "Chimie organică"],
        "subiecte": ["Chimie anorganică", "Chimie organică"],
        "timp_minute": 180,
        "punctaj_total": 100,
    },
    "🧬 Biologie": {
        "cod": "biologie",
        "profile": ["Biologie vegetală și animală", "Anatomie și fiziologie umană"],
        "subiecte": ["Anatomie", "Genetică", "Ecologie"],
        "timp_minute": 180,
        "punctaj_total": 100,
    },
    "🏛️ Istorie": {
        "cod": "istorie",
        "profile": ["Umanist", "Pedagogic", "Teologic"],
        "subiecte": ["Istorie românească", "Istorie universală"],
        "timp_minute": 180,
        "punctaj_total": 100,
    },
    "🌍 Geografie": {
        "cod": "geografie",
        "profile": ["Profiluri umaniste"],
        "subiecte": ["Geografia României", "Geografia Europei", "Geografia lumii"],
        "timp_minute": 180,
        "punctaj_total": 100,
    },
    "💻 Informatică": {
        "cod": "informatica",
        "profile": ["Mate-Info intensiv C++", "Mate-Info intensiv Pascal"],
        "subiecte": ["Algoritmi", "Structuri de date", "Programare"],
        "timp_minute": 180,
        "punctaj_total": 100,
    },
}




def extract_text_from_photo(image_bytes: bytes, materie_label: str) -> str:
    """Extrage textul scris de mână dintr-o fotografie folosind Gemini Vision."""
    try:
        import base64
        key = keys[st.session_state.get("key_index", 0)]
        genai.configure(api_key=key)
        model = genai.GenerativeModel(get_ai_model())  # Folosește funcția de fallback

        img_b64 = base64.b64encode(image_bytes).decode()
        prompt = (
            f"Ești un asistent care transcrie text scris de mână din lucrări de elevi la {materie_label}. "
            f"Transcrie EXACT tot ce este scris în imagine, inclusiv formule, simboluri matematice și calcule. "
            f"Păstrează structura (Subiectul I, II, III dacă există). "
            f"Dacă un cuvânt e greu de citit, transcrie-l cu [?]. "
            f"Nu adăuga nimic, nu corecta nimic — transcrie fidel."
        )

        response = model.generate_content([
            {"mime_type": "image/jpeg", "data": img_b64},
            prompt
        ])
        return response.text.strip()
    except Exception as e:
        log_error("PhotoOCR", str(e))
        return f"[Eroare la citirea pozei: {e}]"


def get_bac_prompt_ai(materie_label, materie_info, profil):
    subiecte_str = ", ".join(materie_info["subiecte"])
    return (
        f"Generează un subiect complet de BAC la {materie_label} ({profil}), "
        f"identic ca structură și dificultate cu subiectele oficiale din România.\n\n"
        f"STRUCTURĂ OBLIGATORIE:\n"
        f"- SUBIECTUL I (30 puncte): 5 itemi tip grilă/răspuns scurt\n"
        f"- SUBIECTUL II (30 puncte): 3-4 probleme de dificultate medie\n"
        f"- SUBIECTUL III (30 puncte): 1-2 probleme complexe / eseu structurat\n"
        f"- 10 puncte din oficiu\n\n"
        f"TEME: {subiecte_str}\n"
        f"TIMP: {materie_info['timp_minute']} minute\n\n"
        f"La final adaugă baremul astfel:\n"
        f"[[BAREM_BAC]]\n"
        f"SUBIECTUL I: [raspunsuri si punctaj]\n"
        f"SUBIECTUL II: [solutii si punctaj]\n"
        f"SUBIECTUL III: [criterii si punctaj]\n"
        f"[[/BAREM_BAC]]"
    )


def get_bac_correction_prompt(materie_label, subiect, raspuns_elev, from_photo=False):
    source_note = (
        "NOTĂ: Răspunsul a fost extras automat dintr-o fotografie a lucrării. "
        "Unele cuvinte pot fi transcrise imperfect din cauza scrisului de mână — "
        "judecă după intenția elevului, nu după eventuale erori de OCR.\n\n"
        if from_photo else ""
    )

    # Reguli de limbaj adaptate materiei
    if "Română" in materie_label:
        lang_rules = (
            "CORECTARE LIMBĂ ROMÂNĂ (OBLIGATORIU — punctaj separat):\n"
            "- Ortografie și punctuație (virgule, punct, ghilimele «»)\n"
            "- Acordul gramatical (subiect-predicat, adjectiv-substantiv)\n"
            "- Folosirea corectă a cratimei, apostrofului\n"
            "- Exprimare clară, coerentă, fără pleonasme sau cacofonii\n"
            "- Registru stilistic adecvat eseului de BAC\n"
            "- Acordă până la 10 puncte bonus/penalizare pentru calitatea limbii\n\n"
        )
    else:
        lang_rules = (
            f"CORECTARE LIMBAJ ȘTIINȚIFIC ({materie_label}):\n"
            "- Terminologie specifică folosită corect\n"
            "- Notații și simboluri respectate (ex: m pentru masă, nu M; v nu V pentru viteză)\n"
            "- Unități de măsură scrise corect și complet\n"
            "- Formulele scrise corect, fără ambiguități\n"
            "- Raționament logic și coerent exprimat în cuvinte\n"
            "- Acordă până la 5 puncte bonus/penalizare pentru calitatea exprimării\n\n"
        )

    return (
        f"Ești examinator BAC România pentru {materie_label}.\n\n"
        f"{source_note}"
        f"SUBIECTUL:\n{subiect}\n\n"
        f"RĂSPUNSUL ELEVULUI:\n{raspuns_elev}\n\n"
        f"Corectează COMPLET în această ordine:\n\n"
        f"## 📊 Punctaj per subiect\n"
        f"- Subiectul I: X/30 puncte\n"
        f"- Subiectul II: X/30 puncte\n"
        f"- Subiectul III: X/30 puncte\n"
        f"- Din oficiu: 10 puncte\n\n"
        f"## ✅ Ce a făcut bine\n"
        f"[aspecte corecte]\n\n"
        f"## ❌ Greșeli și explicații\n"
        f"[fiecare greșeală explicată]\n\n"
        f"## 🖊️ Calitatea limbii și exprimării\n"
        f"{lang_rules}"
        f"## 🎓 Nota finală\n"
        f"**Nota: X/10** — [verdict scurt]\n\n"
        f"## 💡 Recomandări pentru BAC\n"
        f"[2-3 sfaturi concrete]\n\n"
        f"Fii constructiv, cald, dar riguros ca un examinator real."
    )


def parse_bac_subject(response):
    barem = ""
    subject_text = response
    match = re.search(r"\[\[BAREM_BAC\]\](.*?)\[\[/BAREM_BAC\]\]", response, re.DOTALL)
    if match:
        barem = match.group(1).strip()
        subject_text = response[:match.start()].strip()
    return subject_text, barem


def format_timer(seconds_remaining):
    h = seconds_remaining // 3600
    m = (seconds_remaining % 3600) // 60
    s = seconds_remaining % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def run_bac_sim_ui():
    st.subheader("🎓 Simulare BAC")

    # ── ECRAN DE START ──
    if not st.session_state.get("bac_active"):
        st.markdown(
            "<div style='background:linear-gradient(135deg,#667eea,#764ba2);"
            "color:white;padding:20px 24px;border-radius:12px;margin-bottom:20px'>"
            "<h4 style='margin:0 0 8px 0'>📋 Cum funcționează?</h4>"
            "<ul style='margin:0;padding-left:18px;line-height:1.8'>"
            "<li>Alegi materia, profilul și tipul de subiect</li>"
            "<li>Rezolvi în timp real cu cronometru opțional</li>"
            "<li>Primești corectare AI detaliată + barem</li>"
            "</ul></div>",
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        with col1:
            bac_materie = st.selectbox("📚 Materia:", options=list(MATERII_BAC.keys()), key="bac_mat_sel")
            info = MATERII_BAC[bac_materie]
            bac_profil = st.selectbox("🎯 Profil:", options=info["profile"], key="bac_prof_sel")
        with col2:
            bac_tip = "🤖 Generat de AI"
            use_timer = st.checkbox(f"⏱️ Cronometru ({info['timp_minute']} min)", value=True, key="bac_timer")


        st.divider()
        col_s, col_b = st.columns(2)
        with col_s:
            btn_lbl = "🚀 Generează subiect AI"
            if st.button(btn_lbl, type="primary", use_container_width=True):
                if "AI" in bac_tip:
                    with st.spinner("📝 Se generează subiectul BAC..."):
                        prompt = get_bac_prompt_ai(bac_materie, info, bac_profil)
                        full = "".join(run_chat_with_rotation(
                            [], [prompt],
                            system_prompt=get_system_prompt(MATERII.get(bac_materie))
                        ))
                    subject_text, barem = parse_bac_subject(full)


                st.session_state.update({
                    "bac_active": True,
                    "bac_materie": bac_materie,
                    "bac_profil": bac_profil,
                    "bac_tip": bac_tip,
                    "bac_subject": subject_text,
                    "bac_barem": barem,
                    "bac_raspuns": "",
                    "bac_corectat": False,
                    "bac_corectare": "",
                    "bac_start_time": time.time() if use_timer else None,
                    "bac_timp_min": info["timp_minute"],
                    "bac_use_timer": use_timer,
                })
                st.rerun()
        with col_b:
            if st.button("↩️ Înapoi la chat", use_container_width=True):
                st.session_state.pop("bac_mode", None)
                st.rerun()
        return

    # ── SIMULARE ACTIVĂ ──
    col_title, col_timer = st.columns([3, 1])
    with col_title:
        st.markdown(f"### {st.session_state.bac_materie} · {st.session_state.bac_profil}")
    with col_timer:
        if st.session_state.get("bac_use_timer") and st.session_state.get("bac_start_time"):
            elapsed = int(time.time() - st.session_state.bac_start_time)
            total   = st.session_state.bac_timp_min * 60
            left    = max(0, total - elapsed)
            pct     = left / total
            color   = "#2ecc71" if pct > 0.5 else ("#e67e22" if pct > 0.2 else "#e74c3c")
            st.markdown(
                f'<div style="background:{color};color:white;padding:8px 12px;'
                f'border-radius:8px;text-align:center;font-size:20px;font-weight:700">'
                f'⏱️ {format_timer(left)}</div>',
                unsafe_allow_html=True
            )
            if left == 0:
                st.warning("⏰ Timpul a expirat!")

    st.divider()

    with st.expander("📋 Subiectul", expanded=not st.session_state.bac_corectat):
        st.markdown(st.session_state.bac_subject)

    if not st.session_state.bac_corectat:
        st.markdown("### ✏️ Răspunsurile tale")

        tab_foto, tab_text = st.tabs(["📷 Fotografiază lucrarea", "⌨️ Scrie manual"])

        raspuns = st.session_state.get("bac_raspuns", "")
        from_photo = False

        # ── TAB FOTO ──
        with tab_foto:
            st.info(
                "📱 **Pe telefon:** apasă butonul de mai jos și fotografiază lucrarea.\n\n"
                "💻 **Pe calculator:** încarcă o poză din galerie.\n\n"
                "AI-ul va citi textul și va porni corectarea automat."
            )
            uploaded_photo = st.file_uploader(
                "Încarcă fotografia lucrării:",
                type=["jpg", "jpeg", "png", "webp", "heic"],
                key="bac_photo_upload",
                help="Fă o poză clară, cu lumină bună, la lucrarea scrisă de mână."
            )

            if uploaded_photo:
                st.image(uploaded_photo, caption="Fotografia încărcată", use_container_width=True)

                if not st.session_state.get("bac_ocr_done"):
                    with st.spinner("🔍 Profesorul citește lucrarea..."):
                        img_bytes = uploaded_photo.read()
                        text_extras = extract_text_from_photo(img_bytes, st.session_state.bac_materie)
                    st.session_state.bac_raspuns  = text_extras
                    st.session_state.bac_ocr_done = True
                    st.session_state.bac_from_photo = True

                    # Pornește corectura automat
                    with st.spinner("📊 Se corectează lucrarea..."):
                        prompt = get_bac_correction_prompt(
                            st.session_state.bac_materie,
                            st.session_state.bac_subject,
                            text_extras,
                            from_photo=True
                        )
                        corectare = "".join(run_chat_with_rotation(
                            [], [prompt],
                            system_prompt=get_system_prompt(MATERII.get(st.session_state.bac_materie))
                        ))
                    st.session_state.bac_corectare = corectare
                    st.session_state.bac_corectat  = True
                    st.rerun()

                if st.session_state.get("bac_ocr_done"):
                    with st.expander("📄 Text extras din poză", expanded=False):
                        st.text(st.session_state.get("bac_raspuns", ""))

        # ── TAB TEXT ──
        with tab_text:
            raspuns = st.text_area(
                "Scrie rezolvarea completă:",
                value=st.session_state.get("bac_raspuns", ""),
                height=350,
                placeholder="Subiectul I:\n1. ...\n2. ...\n\nSubiectul II:\n...\n\nSubiectul III:\n...",
                key="bac_ans_input"
            )
            st.session_state.bac_raspuns = raspuns
            st.session_state.bac_from_photo = False

            if st.button("🤖 Corectare AI", type="primary", use_container_width=True,
                         disabled=not raspuns.strip()):
                with st.spinner("📊 Se corectează lucrarea..."):
                    prompt = get_bac_correction_prompt(
                        st.session_state.bac_materie,
                        st.session_state.bac_subject,
                        raspuns,
                        from_photo=False
                    )
                    corectare = "".join(run_chat_with_rotation(
                        [], [prompt],
                        system_prompt=get_system_prompt(MATERII.get(st.session_state.bac_materie))
                    ))
                st.session_state.bac_corectare = corectare
                st.session_state.bac_corectat  = True
                st.rerun()

        st.divider()
        col_barem, col_nou = st.columns(2)
        with col_barem:
            if st.session_state.get("bac_barem"):
                if st.button("📋 Arată Baremul", use_container_width=True):
                    st.session_state.bac_show_barem = not st.session_state.get("bac_show_barem", False)
                    st.rerun()
        with col_nou:
            if st.button("🔄 Subiect nou", use_container_width=True):
                for k in [k for k in list(st.session_state.keys()) if k.startswith("bac_")]:
                    st.session_state.pop(k, None)
                st.rerun()

        if st.session_state.get("bac_show_barem") and st.session_state.get("bac_barem"):
            with st.expander("📋 Barem de corectare", expanded=True):
                st.markdown(st.session_state.bac_barem)

    else:
        st.markdown("### 📊 Corectare AI")
        st.markdown(st.session_state.bac_corectare)
        if st.session_state.get("bac_barem"):
            with st.expander("📋 Barem"):
                st.markdown(st.session_state.bac_barem)
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Subiect nou", type="primary", use_container_width=True):
                for k in [k for k in list(st.session_state.keys()) if k.startswith("bac_")]:
                    st.session_state.pop(k, None)
                st.rerun()
        with col2:
            if st.button("✏️ Reîncerc același subiect", use_container_width=True):
                st.session_state.bac_corectat  = False
                st.session_state.bac_corectare = ""
                st.session_state.bac_raspuns   = ""
                if st.session_state.get("bac_use_timer"):
                    st.session_state.bac_start_time = time.time()
                st.rerun()
        with col3:
            if st.button("💬 Înapoi la chat", use_container_width=True):
                for k in [k for k in list(st.session_state.keys()) if k.startswith("bac_")]:
                    st.session_state.pop(k, None)
                st.session_state.pop("bac_mode", None)
                st.rerun()


# === MOD QUIZ ===
NIVELE_QUIZ = ["🟢 Ușor (gimnaziu)", "🟡 Mediu (liceu)", "🔴 Greu (BAC)"]

MATERII_QUIZ = [m for m in list(MATERII.keys()) if m != "🎓 Toate materiile"]


def get_quiz_prompt(materie_label: str, nivel: str, materie_val: str) -> str:
    """Generează prompt pentru crearea unui quiz."""
    nivel_text = nivel.split(" ", 1)[1].strip("()")
    return f"""Generează un quiz de 5 întrebări la {materie_label} pentru nivel {nivel_text}.

REGULI STRICTE:
1. Generează EXACT 5 întrebări numerotate (1. 2. 3. 4. 5.)
2. Fiecare întrebare are 4 variante de răspuns: A) B) C) D)
3. La finalul TUTUROR întrebărilor adaugă un bloc special cu răspunsurile corecte:

[[RASPUNSURI_CORECTE]]
1: X
2: X
3: X
4: X
5: X
[[/RASPUNSURI_CORECTE]]

unde X este A, B, C sau D.
4. Întrebările trebuie să fie clare și potrivite pentru nivel {nivel_text}.
5. Folosește LaTeX ($...$) pentru formule matematice.
6. NU da explicații acum — doar întrebările și răspunsurile corecte la final."""


def parse_quiz_response(response: str) -> tuple[str, dict]:
    """Extrage întrebările și răspunsurile corecte din răspunsul AI."""
    correct = {}
    clean_response = response
    
    match = re.search(r'\[\[RASPUNSURI_CORECTE\]\](.*?)\[\[/RASPUNSURI_CORECTE\]\]',
                      response, re.DOTALL)
    if match:
        clean_response = response[:match.start()].strip()
        for line in match.group(1).strip().splitlines():
            line = line.strip()
            if ':' in line:
                parts = line.split(':', 1)
                try:
                    q_num = int(parts[0].strip())
                    ans = parts[1].strip().upper()
                    if ans in ['A', 'B', 'C', 'D']:
                        correct[q_num] = ans
                except ValueError:
                    continue
    else:
        # Încercăm să găsim răspunsurile în alt format
        st.warning("⚠️ Formatul răspunsurilor nu a fost recunoscut. Verifică generarea quiz-ului.")
        log_error("QuizParse", "Format răspunsuri nerecunoscut", st.session_state.get("session_id"))
    
    return clean_response, correct


def evaluate_quiz(user_answers: dict, correct_answers: dict) -> tuple[int, str]:
    """Evaluează răspunsurile și returnează (scor, feedback_text)."""
    score = sum(1 for q, a in user_answers.items() if correct_answers.get(q) == a)
    total = len(correct_answers)

    lines = []
    for q in sorted(correct_answers.keys()):
        user_ans = user_answers.get(q, "—")
        correct_ans = correct_answers[q]
        if user_ans == correct_ans:
            lines.append(f"✅ **Întrebarea {q}**: {user_ans} — Corect!")
        else:
            lines.append(f"❌ **Întrebarea {q}**: ai răspuns **{user_ans}**, corect era **{correct_ans}**")

    if score == total:
        verdict = "🏆 Excelent! Nota 10!"
    elif score >= total * 0.8:
        verdict = "🌟 Foarte bine!"
    elif score >= total * 0.6:
        verdict = "👍 Bine, mai exersează puțin!"
    elif score >= total * 0.4:
        verdict = "📚 Trebuie să mai studiezi."
    else:
        verdict = "💪 Nu-ți face griji, încearcă din nou!"

    feedback = f"### Rezultat: {score}/{total} — {verdict}\n\n" + "\n\n".join(lines)
    return score, feedback


def run_quiz_ui():
    """Randează UI-ul pentru modul Quiz."""
    st.subheader("📝 Mod Examinare")

    # --- Setup quiz ---
    if not st.session_state.get("quiz_active"):
        col1, col2 = st.columns(2)
        with col1:
            quiz_materie_label = st.selectbox(
                "Materie:",
                options=MATERII_QUIZ,
                key="quiz_materie_select"
            )
        with col2:
            quiz_nivel = st.selectbox(
                "Nivel:",
                options=NIVELE_QUIZ,
                key="quiz_nivel_select"
            )

        if st.button("🚀 Generează Quiz", type="primary", use_container_width=True):
            quiz_materie_val = MATERII[quiz_materie_label]
            with st.spinner("📝 Profesorul pregătește întrebările..."):
                prompt = get_quiz_prompt(quiz_materie_label, quiz_nivel, quiz_materie_val)
                full_resp = ""
                for chunk in run_chat_with_rotation(
                    [], [prompt],
                    system_prompt=get_system_prompt(quiz_materie_val)
                ):
                    full_resp += chunk

            questions_text, correct = parse_quiz_response(full_resp)
            if len(correct) >= 3:
                st.session_state.quiz_active = True
                st.session_state.quiz_questions = questions_text
                st.session_state.quiz_correct = correct
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.quiz_materie = quiz_materie_label
                st.session_state.quiz_nivel = quiz_nivel
                st.rerun()
            else:
                st.error("❌ Nu am putut genera quiz-ul. Încearcă din nou.")
        return

    # --- Quiz activ ---
    st.caption(f"📚 {st.session_state.quiz_materie} · {st.session_state.quiz_nivel}")

    # Afișează întrebările
    st.markdown(st.session_state.quiz_questions)
    st.divider()

    if not st.session_state.quiz_submitted:
        st.markdown("**Alege răspunsurile tale:**")
        answers = {}
        for q_num in sorted(st.session_state.quiz_correct.keys()):
            answers[q_num] = st.radio(
                f"Întrebarea {q_num}:",
                options=["A", "B", "C", "D"],
                horizontal=True,
                key=f"quiz_ans_{q_num}",
                index=None
            )

        all_answered = all(v is not None for v in answers.values())

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Trimite răspunsurile", type="primary",
                         disabled=not all_answered, use_container_width=True):
                st.session_state.quiz_answers = {k: v for k, v in answers.items() if v}
                st.session_state.quiz_submitted = True
                st.rerun()
        with col2:
            if st.button("🔄 Quiz nou", use_container_width=True):
                for k in ["quiz_active", "quiz_questions", "quiz_correct",
                          "quiz_answers", "quiz_submitted"]:
                    st.session_state.pop(k, None)
                st.rerun()
    else:
        # Afișează rezultatele
        score, feedback = evaluate_quiz(
            st.session_state.quiz_answers,
            st.session_state.quiz_correct
        )
        st.markdown(feedback)
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Quiz nou", type="primary", use_container_width=True):
                for k in ["quiz_active", "quiz_questions", "quiz_correct",
                          "quiz_answers", "quiz_submitted"]:
                    st.session_state.pop(k, None)
                st.rerun()
        with col2:
            if st.button("💬 Înapoi la chat", use_container_width=True):
                for k in ["quiz_active", "quiz_questions", "quiz_correct",
                          "quiz_answers", "quiz_submitted", "quiz_mode"]:
                    st.session_state.pop(k, None)
                st.rerun()


def run_chat_with_rotation(history_obj, payload, system_prompt=None):
    """Rulează chat cu rotație automată a cheilor API și fallback model."""
    active_prompt = system_prompt or st.session_state.get("system_prompt") or SYSTEM_PROMPT
    max_retries = len(keys) * 2
    last_error = None
    
    for attempt in range(max_retries):
        try:
            if st.session_state.key_index >= len(keys):
                st.session_state.key_index = 0
            current_key = keys[st.session_state.key_index]
            genai.configure(api_key=current_key)
            
            # Folosește funcția de fallback pentru model
            model_name = get_ai_model()
            
            model = genai.GenerativeModel(
                model_name,
                system_instruction=active_prompt,
                safety_settings=safety_settings
            )
            chat = model.start_chat(history=history_obj)
            response_stream = chat.send_message(payload, stream=True)
            
            for chunk in response_stream:
                try:
                    if chunk.text:
                        yield chunk.text
                except ValueError:
                    continue
            
            # Dacă am ajuns aici, totul e ok - ieșim din funcție
            return
            
        except Exception as e:
            last_error = e
            error_msg = str(e)
            
            if "503" in error_msg or "overloaded" in error_msg:
                st.toast("🐢 Reîncerc...", icon="⏳")
                time.sleep(2)
                continue
            elif "400" in error_msg or "429" in error_msg or "Quota" in error_msg or "API key not valid" in error_msg:
                st.toast(f"⚠️ Schimb cheia {st.session_state.key_index + 1}...", icon="🔄")
                st.session_state.key_index = (st.session_state.key_index + 1) % len(keys)
                continue
            else:
                # Loghează eroarea necunoscută
                log_error("ChatRotation", str(e), st.session_state.get("session_id"))
                raise e
    
    # Dacă am epuizat încercările
    error_msg = f"Serviciul este indisponibil momentan. Ultima eroare: {last_error}"
    log_error("ChatRotation", error_msg, st.session_state.get("session_id"))
    raise Exception(error_msg)


# === UI PRINCIPAL ===
st.title("🎓 Profesor Liceu")

with st.sidebar:
    st.header("⚙️ Opțiuni")

    # --- Selector materie ---
    st.subheader("📚 Materie")
    materie_label = st.selectbox(
        "Alege materia:",
        options=list(MATERII.keys()),
        index=0,
        label_visibility="collapsed"
    )
    materie_selectata = MATERII[materie_label]

    # Actualizează system prompt dacă s-a schimbat materia
    if st.session_state.get("materie_selectata") != materie_selectata:
        st.session_state.materie_selectata = materie_selectata
        st.session_state.system_prompt = get_system_prompt(materie_selectata)

    if materie_selectata:
        st.info(f"Focusat pe: **{materie_label}**")

    st.divider()

    # --- Dark Mode toggle ---
    dark_mode = st.toggle("🌙 Mod Întunecat", value=st.session_state.get("dark_mode", False))
    if dark_mode != st.session_state.get("dark_mode", False):
        st.session_state.dark_mode = dark_mode
        st.rerun()

    st.divider()

    # --- Status Supabase ---
    if not st.session_state.get("_sb_online", True):
        st.markdown(
            '<div style="background:#e67e22;color:white;padding:8px 12px;'
            'border-radius:8px;font-size:13px;text-align:center;margin-bottom:8px">'
            '📴 Mod offline — datele sunt salvate local</div>',
            unsafe_allow_html=True
        )
    else:
        pending = len(st.session_state.get("_offline_queue", []))
        if pending:
            st.caption(f"☁️ {pending} mesaje în așteptare pentru sincronizare")


    st.divider()

    if st.button("🗑️ Șterge Istoricul", type="primary"):
        clear_history_db(st.session_state.session_id)
        st.session_state.messages = []
        invalidate_session_cache()  # Invalidează cache-ul
        st.rerun()

    enable_audio = st.checkbox("🔊 Voce", value=False)

    if enable_audio:
        voice_option = st.radio(
            "🎙️ Alege vocea:",
            options=["👨 Domnul Profesor (Emil)", "👩 Doamna Profesoară (Alina)"],
            index=0
        )
        selected_voice = VOICE_MALE_RO if "Emil" in voice_option else VOICE_FEMALE_RO
    else:
        selected_voice = VOICE_MALE_RO

    st.divider()

    st.header("📁 Materiale")
    uploaded_file = st.file_uploader("Încarcă Poză sau PDF", type=["jpg", "jpeg", "png", "pdf"])
    media_content = None

    if uploaded_file:
        genai.configure(api_key=keys[st.session_state.key_index])
        file_type = uploaded_file.type

        if "image" in file_type:
            media_content = Image.open(uploaded_file)
            st.image(media_content, caption="Imagine atașată", use_container_width=True)
        elif "pdf" in file_type:
            st.info("📄 PDF Detectat. Se procesează...")
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                with st.spinner("📚 Se trimite cartea la AI..."):
                    uploaded_pdf = genai.upload_file(tmp_path, mime_type="application/pdf")
                    while uploaded_pdf.state.name == "PROCESSING":
                        time.sleep(1)
                        uploaded_pdf = genai.get_file(uploaded_pdf.name)
                    media_content = uploaded_pdf
                    st.success(f"✅ Gata: {uploaded_file.name}")
            except Exception as e:
                log_error("PDFUpload", str(e), st.session_state.session_id)
                st.error(f"Eroare upload PDF: {e}")

    st.divider()

    # --- Mod Quiz + BAC ---
    st.divider()
    st.subheader("📝 Examinare & BAC")

    col_q, col_b = st.columns(2)
    with col_q:
        if st.button("🎯 Quiz rapid",
                     use_container_width=True,
                     type="primary" if st.session_state.get("quiz_mode") else "secondary"):
            st.session_state.quiz_mode = not st.session_state.get("quiz_mode", False)
            st.session_state.pop("bac_mode", None)
            for k in list(st.session_state.keys()):
                if k.startswith("bac_"):
                    st.session_state.pop(k, None)
            if not st.session_state.quiz_mode:
                for k in ["quiz_active", "quiz_questions", "quiz_correct",
                          "quiz_answers", "quiz_submitted"]:
                    st.session_state.pop(k, None)
            st.rerun()
    with col_b:
        if st.button("🎓 Simulare BAC",
                     use_container_width=True,
                     type="primary" if st.session_state.get("bac_mode") else "secondary"):
            st.session_state.bac_mode = not st.session_state.get("bac_mode", False)
            st.session_state.pop("quiz_mode", None)
            for k in ["quiz_active", "quiz_questions", "quiz_correct",
                      "quiz_answers", "quiz_submitted"]:
                st.session_state.pop(k, None)
            if not st.session_state.bac_mode:
                for k in list(st.session_state.keys()):
                    if k.startswith("bac_"):
                        st.session_state.pop(k, None)
            st.rerun()

    st.divider()

    # --- Istoric conversații ---
    st.subheader("🕐 Conversații anterioare")
    if st.button("🔄 Conversație nouă", use_container_width=True):
        new_sid = generate_unique_session_id()
        register_session(new_sid)
        switch_session(new_sid)
        st.rerun()

    sessions = get_session_list(limit=15)
    current_sid = st.session_state.session_id
    for s in sessions:
        is_current = s["session_id"] == current_sid
        label = f"{'▶ ' if is_current else ''}{s['preview']}"
        caption = f"{format_time_ago(s['last_active'])} · {s['msg_count']} mesaje"
        with st.container():
            col_btn, col_del = st.columns([5, 1])
            with col_btn:
                if st.button(
                    label,
                    key=f"sess_{s['session_id']}",
                    use_container_width=True,
                    type="primary" if is_current else "secondary",
                    help=caption,
                ):
                    if not is_current:
                        switch_session(s["session_id"])
                        st.rerun()
            with col_del:
                if st.button("🗑", key=f"del_{s['session_id']}", help="Șterge"):
                    clear_history_db(s["session_id"])
                    if is_current:
                        st.session_state.messages = []
                    invalidate_session_cache()  # Invalidează cache-ul
                    st.rerun()

    st.divider()

    if st.checkbox("🔧 Debug Info", value=False):
        msg_count = len(st.session_state.get("messages", []))
        st.caption(f"📊 Mesaje în memorie: {msg_count}/{MAX_MESSAGES_IN_MEMORY}")
        st.caption(f"🔑 Cheie API activă: {st.session_state.key_index + 1}/{len(keys)}")
        st.caption(f"🆔 Sesiune: {st.session_state.session_id[:16]}...")
        st.caption(f"🤖 Model AI: {st.session_state.get('ai_model', AVAILABLE_MODELS[0])}")
        st.caption(f"🎤 TTS azi: {st.session_state.get('tts_count_today', 0)}/50")


# === MAIN UI — BAC / QUIZ / CHAT ===
if st.session_state.get("bac_mode"):
    run_bac_sim_ui()
    st.stop()

if st.session_state.get("quiz_mode"):
    run_quiz_ui()
    st.stop()

# === ÎNCĂRCARE MESAJE (CHAT MODE) ===
if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = load_history_from_db(st.session_state.session_id)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_message_with_svg(msg["content"])
        else:
            st.markdown(msg["content"])


# === CHAT INPUT ===
if user_input := st.chat_input("Întreabă profesorul..."):

    # --- Debounce: blochează mesaje duplicate trimise rapid ---
    now_ts = time.time()
    last_msg = st.session_state.get("_last_user_msg", "")
    last_ts  = st.session_state.get("_last_msg_ts", 0)
    DEBOUNCE_SECONDS = 2.5

    if user_input.strip() == last_msg.strip() and (now_ts - last_ts) < DEBOUNCE_SECONDS:
        st.toast("⏳ Mesaj duplicat ignorat.", icon="🔁")
        st.stop()

    st.session_state["_last_user_msg"] = user_input
    st.session_state["_last_msg_ts"]  = now_ts

    
    context_messages = get_context_for_ai(st.session_state.messages)
    history_obj = []
    for msg in context_messages:
        role_gemini = "model" if msg["role"] == "assistant" else "user"
        history_obj.append({"role": role_gemini, "parts": [msg["content"]]})
    
    final_payload = []
    if media_content:
        final_payload.append("Analizează materialul atașat:")
        final_payload.append(media_content)
    final_payload.append(user_input)
    
    TYPING_HTML = """
    <div class="typing-indicator">
        <div class="typing-dots"><span></span><span></span><span></span></div>
        <span>Domnul Profesor scrie...</span>
    </div>
    """

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Typing indicator înainte să înceapă streaming-ul
        message_placeholder.markdown(TYPING_HTML, unsafe_allow_html=True)

        try:
            stream_generator = run_chat_with_rotation(history_obj, final_payload)
            first_chunk = True

            for text_chunk in stream_generator:
                full_response += text_chunk
                if first_chunk:
                    first_chunk = False  # typing indicator dispare la primul chunk

                if "<svg" in full_response or ("<path" in full_response and "stroke=" in full_response):
                    message_placeholder.markdown(
                        full_response.split("<path")[0] + "\n\n*🎨 Domnul Profesor desenează...*\n\n▌"
                    )
                else:
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.empty()
            render_message_with_svg(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            save_message_with_limits(st.session_state.session_id, "assistant", full_response)
            
            if enable_audio:
                with st.spinner("🎙️ Domnul Profesor vorbește..."):
                    audio_file = generate_professor_voice(full_response, selected_voice)
                    
                    if audio_file:
                        st.audio(audio_file, format='audio/mp3')
                    else:
                        st.caption("🔇 Nu am putut genera vocea pentru acest răspuns.")
                        
        except Exception as e:
            st.error(f"❌ Eroare: {e}")
            log_error("Chat", str(e), st.session_state.session_id)
