"""Generate synthetic Indian vishing / non-vishing call transcripts via the Groq API.

Appends to data/synthetic_conversations.jsonl, continuing conversation/pair id numbering.
Safe to interrupt and re-run: every record is flushed as it is produced, and a re-run
repairs any contrastive pair left half-finished by a previous run.

  python generate_synthetic_data_groq.py --count 1126
  python generate_synthetic_data_groq.py --target 1340 --workers 8
"""

import argparse
import json
import os
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "synthetic_conversations.jsonl")

MAX_RETRIES = 5
TEMPERATURE = 1.0
MAX_COMPLETION_TOKENS = 6000
DEFAULT_WORKERS = 4

DEFAULT_MODELS = {"vertex": "gemini-2.5-flash", "groq": "openai/gpt-oss-120b"}

# Set by init_provider(); the rest of the script never talks to an SDK directly.
PROVIDER = "vertex"
MODEL_NAME = DEFAULT_MODELS["vertex"]
_client = None

# Optional per-provider knobs that speed things up but may be rejected; each is dropped
# automatically and permanently the first time the API refuses it.
_fast_mode = threading.Event()
_fast_mode.set()


def init_provider(provider, model=None):
    """Create the API client. Imports are lazy so only the provider in use need be installed."""
    global PROVIDER, MODEL_NAME, _client
    PROVIDER = provider
    MODEL_NAME = model or DEFAULT_MODELS[provider]

    if provider == "vertex":
        from google import genai
        key = os.environ.get("VERTEX_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise SystemExit("Set VERTEX_API_KEY (see ~/.config/digital_arrest/vertex.env)")
        _client = genai.Client(vertexai=True, api_key=key)
    elif provider == "groq":
        from groq import Groq
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise SystemExit("Set GROQ_API_KEY")
        _client = Groq(api_key=key)
    else:
        raise SystemExit(f"unknown provider {provider!r}")

TACTIC_VOCAB = ["authority", "fear", "urgency", "threat", "isolation", "financial_coercion"]
SCAM_CLASSES = ("digital_arrest_scam", "other_scam")

DA_ORGS = ["CBI", "Mumbai Police Cyber Cell", "Delhi Police Cyber Cell", "TRAI", "Customs",
           "RBI", "Income Tax Department", "Narcotics Control Bureau", "Bank fraud desk"]
DA_PRETEXTS = ["parcel with contraband", "SIM card misuse", "money laundering case",
               "Aadhaar misuse", "fake case registered in your name", "KYC expiry",
               "income tax notice", "bank account freeze"]
VICTIM_PERSONAS = ["elderly person with low digital literacy", "young working professional",
                   "homemaker", "college student", "retired government officer",
                   "small business owner", "school teacher", "IT employee",
                   "auto rickshaw driver", "NRI returning to India", "farmer",
                   "army veteran", "bank employee", "government clerk", "shopkeeper",
                   "nurse", "delivery partner", "software engineer"]
CODESWITCH_RATIOS = ["Hindi-dominant with occasional English words",
                     "balanced Hindi-English code-mixing",
                     "English-dominant but with Hindi words woven throughout"]

# Romanized Hindi markers, used to verify every conversation really is code-mixed rather
# than plain English. Deliberately a small high-frequency set: it only has to prove that
# Hindi is present, not measure how much.
# NB: "the", "phone" and "hi" are deliberately absent. Hindi "the" (were) and "hi"
# (emphatic) collide with ordinary English words, which made plain-English conversations
# score as code-mixed.
HINDI_MARKERS = set("""main mai mera meri mujhe hum humein humara aap aapka aapko aapke tum tumhara
hai hain hoon ho tha thi hoga hogi karo kiya karna karne kare kijiye karenge karta karti
nahi nahin na mat kya kyun kaise kahan kab kaun kitna kitni koi kuch sab bahut thoda zyada
abhi turant jaldi phir baad pehle ab aaj kal raat subah shaam din saal waqt samay
paisa paise rupaye lakh hazaar ghar parivaar beta beti bhai behen
ji namaste shukriya dhanyavaad maaf kripya accha achha theek bilkul haan arey
bol bolo bola boli bataye batao bataiye suniye suno dekho dekhiye samajh samjha samjhe
baat baatein matlab lekin aur ya toh bhi se ko ka ki ke par pe mein liye wala wali
dijiye dein lijiye lo le de diya dena mil mile milega chahiye chahta chahti raha rahi rahe
pata yaad darr pareshan zaroori jaroori galat sahi karke chalo ruko""".split())
# Length-independent: a long English call with two stray Hindi words is not code-mixed,
# while a genuine Hinglish call easily uses a dozen distinct markers.
MIN_HINDI_MARKERS = 4

CITIES = ["Delhi", "Mumbai", "Pune", "Lucknow", "Jaipur", "Nagpur", "Indore", "Patna",
          "Bhopal", "Ranchi", "Chandigarh", "Guwahati", "Surat", "Kanpur", "Nashik",
          "Hyderabad", "Bengaluru", "Kochi", "Dehradun", "Varanasi", "Ludhiana", "Raipur"]
CALLER_NAMES = ["Rajesh", "Amit", "Vikram", "Suresh", "Rahul", "Deepak", "Sanjay", "Manoj",
                "Arjun", "Ravi", "Ajay", "Vinod", "Pankaj", "Anil", "Naveen", "Satish",
                "Priya", "Anjali", "Pooja", "Sunita", "Kavita", "Neha", "Rekha", "Divya",
                "Shreya", "Anita", "Swati", "Nisha", "Farhan", "Imran", "Joseph", "Thomas"]
VICTIM_NAMES = ["Arvind", "Sunil", "Kamla", "Jyoti", "Rohan", "Sneha", "Mohan", "Usha",
                "Prakash", "Shanti", "Yogesh", "Lakshmi", "Baldev", "Seema", "Tarun",
                "Kavya", "Harpreet", "Zoya", "Irfan", "Fatima", "Aarav", "Ishaan",
                "Meenakshi", "Gurpreet", "Nandini", "Rupa", "Devendra", "Shabana"]
TIMES_OF_DAY = ["early morning, the recipient has just woken up",
                "mid-morning on a working day",
                "lunch time, there is background noise",
                "late afternoon",
                "evening, the recipient is at home with family around",
                "late at night, the recipient was asleep"]

CALLER_MANNERS = [
    "calm and professional, never raises their voice, which makes the pressure feel routine",
    "aggressive and rushed, interrupts the recipient mid-sentence",
    "overly formal and bureaucratic, keeps quoting section numbers and procedure",
    "friendly and sympathetic at first, then turns cold and clipped when resisted",
    "young and slightly nervous, clearly reading from a script, stumbles once",
    "senior-sounding and condescending, treats the recipient as if they are wasting his time",
    "theatrically helpful, repeatedly says they are trying to protect the recipient",
    "flat and bored, as if making the hundredth call of the day",
]
OPENING_STYLES = [
    "starts mid-flow, as if continuing an earlier call or transferred line",
    "first demands the recipient confirm their full name and date of birth before explaining anything",
    "opens by reciting a case/reference number and nothing else, waiting for a reaction",
    "greets warmly and asks whether this is a good time to talk",
    "states the problem immediately, with no greeting at all",
    "mispronounces the recipient's name, is corrected, then continues",
    "claims to be calling back about a notice supposedly already sent by post or SMS",
    "asks whether the recipient is alone and can talk freely",
]

SCAM_OUTCOMES = [
    "the recipient is frightened into full compliance and makes the payment or shares the OTP by the end",
    "the recipient goes along with most of it, then hesitates at the final step, and the call ends unresolved",
    "the recipient grows suspicious midway, refuses, and says they will check with their family or bank first",
    "the recipient is scared at first but catches a specific inconsistency in the caller's story and challenges it",
    "the recipient hangs up abruptly quite early, before the money is ever discussed",
    "the recipient pretends to cooperate in order to stall, and keeps asking for written proof or an official email",
    "the recipient panics, begs for time, and promises to pay later without actually paying during the call",
    "the recipient involves someone else in the room, who becomes sceptical and ends the call",
]
LEGIT_OUTCOMES = [
    "the recipient is reassured and agrees to complete the formality at the branch or on the official portal",
    "the recipient is suspicious at first because they have heard about such scams, and the caller patiently explains how to verify the call independently",
    "the recipient asks to call back on the official published number instead, and the caller fully supports that",
    "the recipient is confused about the process and needs it explained again in simpler terms",
    "the recipient asks a practical question about documents or deadlines and gets a clear answer",
]
BENIGN_TEXTURES = [
    "a minor complaint or friction comes up and is handled politely",
    "the recipient is busy and asks to be called back later",
    "there is a small misunderstanding at the start which gets cleared up",
    "there is warm small talk about family or an upcoming festival before the practical point",
    "the recipient is distracted and asks the caller to repeat something",
    "a scheduling conflict has to be negotiated between them",
    "the call is brief and purely transactional, both are in a hurry",
    "the caller has to put the recipient on hold briefly to check a detail",
]

# Each other_scam subtype carries its own flavour note so the model does not fall back on
# digital-arrest / FIR / arrest-warrant language for every kind of scam.
OTHER_SCAM_SUBTYPES = {
    "lottery": {
        "org": "Lottery board / KBC lucky draw",
        "pretext": "lottery win",
        "flavour": "Excited, congratulatory tone. Greed plus urgency plus a small 'processing/GST fee' ask. No legal threats, no arrest talk.",
    },
    "job": {
        "org": "HR consultancy",
        "pretext": "job offer",
        "flavour": "Work-from-home job offer. Pressure about limited seats, asks for a supposedly refundable registration or training fee.",
    },
    "loan": {
        "org": "Loan company / fintech app",
        "pretext": "loan approval",
        "flavour": "Pre-approved personal loan. Pressure that the offer expires today, asks for a one-time processing or insurance fee.",
    },
    "courier": {
        "org": "Courier company (FedEx/DHL/Bluedart)",
        "pretext": "parcel with contraband",
        "flavour": "Parcel held at customs with illegal items supposedly found inside. Fear plus a fine or clearance fee. It may edge toward a vague cyber-crime case threat, but must stay distinct from police/FIR digital-arrest phrasing.",
    },
    "otp_bank": {
        "org": "Bank fraud desk",
        "pretext": "card blocked / suspicious transaction",
        "flavour": "Fake urgent fraud alert on a card. Pressure to read out the OTP immediately in order to 'block' a fraudulent transaction.",
    },
    "tech_support": {
        "org": "Microsoft support / Amazon security team",
        "pretext": "virus detected / account compromised",
        "flavour": "Technical, alarmed tone about a hacked device or compromised account. Pushes for remote access or payment for security software.",
    },
    "investment": {
        "org": "Investment advisory group / trading signals group",
        "pretext": "guaranteed stock or crypto returns",
        "flavour": "FOMO-driven. Fake certainty about guaranteed returns, pushes a deposit into a 'trading account' before a price window closes.",
    },
    "electricity": {
        "org": "Electricity board",
        "pretext": "bill overdue, disconnection threat",
        "flavour": "Fake urgent disconnection within hours unless paid immediately by UPI.",
    },
    "sextortion": {
        "org": "Unknown caller",
        "pretext": "intimate video blackmail",
        "flavour": "Cold, threatening tone. Claims to hold a compromising recording and threatens to send it to contacts or family unless paid. Keep it non-graphic: focus on the fear, threat and isolation dynamics, never on explicit content.",
    },
    "matrimonial": {
        "org": "matrimonial site agent",
        "pretext": "profile verification fee",
        "flavour": "Fake profile verification or premium fee, with pressure about losing matches or visibility if it is not paid today.",
    },
    "army_fake": {
        "org": "fake army officer",
        "pretext": "online marketplace deal needing urgent advance payment",
        "flavour": "Sentimental and rushed. A story about being posted at the border with no time or network access, urgently needing an advance transfer.",
    },
}

BENIGN_CALL_TYPES = [
    {"org": "courier/delivery company", "pretext": "parcel delivery confirmation"},
    {"org": "bank customer service", "pretext": "routine KYC update reminder"},
    {"org": "market research agency", "pretext": "customer satisfaction survey"},
    {"org": "doctor's clinic", "pretext": "appointment confirmation reminder"},
    {"org": "school accounts office", "pretext": "fee payment reminder"},
    {"org": "telecom company", "pretext": "plan upgrade / promotional offer"},
    {"org": "insurance company", "pretext": "policy renewal reminder"},
    {"org": "family member or friend", "pretext": "casual check-in / catching up"},
    {"org": "HR team", "pretext": "job interview scheduling"},
    {"org": "landlord", "pretext": "rent and maintenance follow-up"},
    {"org": "home service technician (AC/plumber/electrician)", "pretext": "service appointment confirmation"},
    {"org": "government helpline", "pretext": "pension or passport status callback"},
    {"org": "gym", "pretext": "membership renewal reminder"},
    {"org": "food delivery service", "pretext": "order confirmation / delay update"},
    {"org": "housing society secretary", "pretext": "society meeting coordination"},
    {"org": "school class teacher", "pretext": "parent-teacher meeting scheduling"},
    {"org": "travel agent", "pretext": "train or flight booking confirmation"},
    {"org": "pharmacy", "pretext": "medicine stock and home delivery"},
]

STYLE_GUIDANCE = """
Write genuinely natural dialogue, not a formulaic script:
- People interrupt, repeat themselves, say "haan haan", trail off, and ask things twice.
  Turns should differ in length; some are three words, some are several sentences.
- Do not narrate or label behaviour inside the text. Write only what each person says aloud.
- Use the specific names, city, amounts and reference numbers given above instead of inventing
  generic ones, and weave them in the way real speakers do (half-remembered, misheard, repeated
  back for confirmation).
- Keep every manipulative or persuasive line in the speaker's own words. Never explain the tactic.
"""

TACTIC_GUIDANCE = f"""
Each turn must carry a "tactics" field: a list of zero or more tags from exactly this vocabulary:
{TACTIC_VOCAB}
- Victim turns always get an empty list [].
- A caller turn gets a tag only when that turn actually performs that behaviour. Plain
  conversational caller turns (greetings, acknowledgements, small talk) get [] as well.
- A caller turn may carry two tags when it genuinely blends behaviours, e.g. ["threat","urgency"].
- In a scam call the tagged caller turns should escalate as the call progresses, roughly in the
  order authority -> fear -> urgency -> threat -> isolation -> financial_coercion. Not every stage
  must appear, and the mix must fit the scam type: a lottery scam leans on urgency and
  financial_coercion and rarely threatens, while a blackmail scam leans on fear, threat and isolation.
- In a legitimate_authority_hard_negative call the caller may use only "authority" and mild
  "urgency". Never "fear", "threat", "isolation" or "financial_coercion".
- In a benign call every turn, caller and victim alike, gets [].
"""


# A lottery board or matrimonial agent quoting an "FIR number" reads as fake, so the
# reference format follows whoever the caller is pretending to be.
_REF_STYLES = {
    "law": ["FIR", "CR", "CASE", "DD"],
    "lottery": ["WIN", "DRAW", "KBC"],
    "job": ["APP", "CAND", "REQ"],
    "loan": ["LN", "APP", "SANC"],
    "courier": ["AWB", "CN", "CUS"],
    "otp_bank": ["TXN", "REF", "DISP"],
    "tech_support": ["TKT", "SR", "INC"],
    "investment": ["CL", "ACC", "PMS"],
    "electricity": ["CA", "CONS", "BU"],
    "matrimonial": ["PRF", "MEM", "ID"],
    "sextortion": [],      # a blackmailer cites no paperwork
    "army_fake": [],
}


def _fake_ref(context="law"):
    styles = _REF_STYLES.get(context, _REF_STYLES["law"])
    if not styles:
        return None
    return f"{random.choice(styles)}/{random.randint(2023, 2026)}/{random.randint(1000, 99999)}"


def _fake_upi():
    return f"{random.choice(['verify', 'secure', 'clear', 'settle', 'refund'])}{random.randint(10, 999)}@{random.choice(['okaxis', 'ybl', 'paytm', 'ibl', 'oksbi'])}"


def _amount_for(cls, subtype_key=None):
    if subtype_key in ("lottery", "job", "matrimonial"):
        return random.choice([499, 999, 1200, 1500, 2500, 3500, 4999])
    if subtype_key in ("investment", "loan"):
        return random.choice([15000, 25000, 40000, 50000, 75000, 100000, 150000])
    return random.choice([2000, 3500, 5000, 7500, 9999, 12000, 18000, 25000, 45000, 85000])


def build_prompt(spec):
    lines = [
        "Generate one realistic Indian phone conversation as structured JSON.",
        "",
        f"Caller identity / organisation: {spec['org']}",
        f"Reason for the call: {spec['pretext']}",
        f"Call recipient: {spec['victim_name']}, a {spec['persona']}, in {spec['city']}",
        f"Caller's name: {spec['caller_name']}",
        f"Language style: {spec['codeswitch']}",
        f"When the call happens: {spec['time_of_day']}",
        f"Approximate length: {spec['n_turns']} turns, alternating caller and victim, caller first",
        "",
        "Mandatory variation constraints for this specific conversation:",
        f"- The caller's manner: {spec['manner']}",
        f"- How the call opens: {spec['opening']}",
        f"- How the call must end: {spec['outcome']}",
        "- Script: write every Hindi word in romanized Latin letters, the way Indians type "
        "Hinglish on a phone (\"Namaste, main CBI se bol raha hoon\"). Use ONLY the English "
        "alphabet - no Devanagari, Urdu, Bengali, Gujarati or any other script, not even for "
        "a single word.",
        "- The dialogue must genuinely mix Hindi and English words, not be plain English. "
        "Both speakers should use common Hindi words (main, aap, hai, nahi, kya, abhi, "
        "paisa, theek hai, bataiye) naturally alongside English ones.",
        "- Write identifiers as realistic digits, never masked: say \"Aadhaar number 4821 "
        "7736 9014\" or \"account 50100294837261\", never \"XXXX XXXX 9014\" or \"98765-XXXXX\".",
        "- Output only spoken words. No stage directions, no narration, no parentheticals "
        "like (pause), (laughs) or (background noise).",
    ]
    if spec.get("amount"):
        lines.append(f"- Money discussed (if money comes up at all): around Rs. {spec['amount']:,}")
    if spec.get("ref_id"):
        lines.append(f"- Reference/case number the caller cites: {spec['ref_id']}")
    if spec.get("upi"):
        lines.append(f"- Payment handle the caller pushes (only if the call reaches that stage): {spec['upi']}")
    lines.append("")

    cls = spec["class"]
    if cls == "digital_arrest_scam":
        lines.append(
            "This is a SCAM call of the 'digital arrest' kind. The caller falsely claims to be an "
            "official, cites a fabricated case, and works to frighten the recipient into paying money "
            "or sharing an OTP, while pressing them to stay on the line and tell nobody. The caller is "
            "a criminal pretending to hold authority: the authority is fake and the case does not exist."
        )
    elif cls == "legitimate_authority_hard_negative":
        lines.append(
            "This call is NOT a scam. The caller genuinely works for the organisation above and is "
            "calling about a real, mundane administrative matter. It should sound similarly formal and "
            "official to a scam call, and may mention a genuine consequence or deadline, but it must NOT "
            "contain any of the following:\n"
            "- any demand for secrecy or isolation, such as telling the recipient not to tell anyone or "
            "not to hang up\n"
            "- any request for an OTP, PIN, password, card number, or a transfer to a personal "
            "account or UPI handle\n"
            "It MUST include at least one thing a real institution does and a scammer never does, for "
            "example telling the recipient they can call back on the number published on the official "
            "website, or that they may come to the branch in person, or that the organisation will never "
            "ask for a PIN or OTP over the phone."
        )
    elif cls == "benign":
        lines.append(
            "This is an ordinary, non-scam, non-official everyday phone call about the reason given "
            "above. Nothing to do with police, arrest, fraud, or any urgent legal or financial threat. "
            "Nobody is manipulating anybody."
        )
    elif cls == "other_scam":
        lines.append(
            "This IS a scam call, but NOT the digital-arrest kind. The specific flavour of this scam:\n"
            f"{spec['flavour']}\n"
            "Follow that flavour rather than defaulting to police, FIR or arrest language."
        )

    lines.append(STYLE_GUIDANCE)
    lines.append(TACTIC_GUIDANCE)
    lines.append(
        'Output STRICT JSON only, no prose and no markdown fences, exactly in this shape:\n'
        '{"turns": [{"turn_id": 0, "speaker": "caller", "text": "...", "tactics": ["authority"]}, '
        '{"turn_id": 1, "speaker": "victim", "text": "...", "tactics": []}]}'
    )
    return "\n".join(lines)


# A few words the model reaches for in the wrong script so persistently that retrying is
# wasted effort ("ke तहत" survived five regenerations). Only words whose romanization has
# been checked in context are listed; anything else is caught by validation and retried.
STUBBORN_TRANSLITERATIONS = {
    "तहत": "tahat", "मामला": "maamla", "तुरंत": "turant", "नहीं": "nahi",
    "पैसे": "paise", "पैसा": "paisa", "अभी": "abhi", "तक": "tak",
    "معاملہ": "maamla", "جلدی": "jaldi", "لেনদেন": "lenden",
    "કાર્યવાહી": "karyavahi", "নিশ্চিত": "nishchit",
}


def transliterate_stubborn(text):
    for src, dst in STUBBORN_TRANSLITERATIONS.items():
        if src in text:
            text = text.replace(src, dst)
    return text


def normalize_turns(cls, turns):
    """Coerce the model's turns into the dataset schema. Raises ValueError if unusable."""
    if not isinstance(turns, list) or len(turns) < 4:
        raise ValueError(f"expected a list of at least 4 turns, got {type(turns).__name__} "
                         f"of length {len(turns) if isinstance(turns, list) else 'n/a'}")
    cleaned = []
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            raise ValueError(f"turn {i} is not an object")
        speaker = str(t.get("speaker", "")).strip().lower()
        if speaker not in ("caller", "victim"):
            raise ValueError(f"turn {i} has unusable speaker {t.get('speaker')!r}")
        text = transliterate_stubborn(str(t.get("text", "")).strip())
        if not text:
            raise ValueError(f"turn {i} has empty text")
        tags = t.get("tactics") or []
        if isinstance(tags, str):
            tags = [tags]
        tags = [g for g in (str(x).strip().lower() for x in tags) if g in TACTIC_VOCAB]
        if speaker == "victim" or cls == "benign":
            tags = []
        cleaned.append({"turn_id": i, "speaker": speaker, "text": text,
                        "tactics": sorted(set(tags), key=TACTIC_VOCAB.index)})
    return cleaned


_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
# Zero tolerance. These are not Hindi deliberately written in Hindi, they are the model
# glitching mid-word - "mujें" for "mujhe", "ke तहत" for "ke tahat", and the same thing in
# sibling scripts: "معاملہ" for "maamla", "লেনদেন" for "lenden", "કાર્યવાહી" for
# "karyavahi". All of it tokenises badly, so any non-Latin letter fails validation.
MAX_NON_LATIN_CHARS = 0


def non_latin_chars(turns):
    """Count alphabetic characters outside the Latin blocks (Devanagari, Bengali, Arabic...)."""
    text = " ".join(t["text"] for t in turns)
    return sum(1 for ch in text if ch.isalpha() and ord(ch) > 0x024F)


def devanagari_chars(turns):
    return len(_DEVANAGARI.findall(" ".join(t["text"] for t in turns)))


_MASKED_DIGITS = re.compile(r"[xX*#]{3,}|\bX{2,}[\s-]?X{2,}", re.ASCII)
_NARRATION = re.compile(
    r"\((?:laughs?|laughing|pause|pauses|sighs?|crying|sobs?|silence|background|aside|"
    r"to (?:another|someone|himself|herself)|whisper\w*|inaudible|muffled|noise)[^)]*\)"
    r"|\[(?:pause|laughs?|silence|inaudible|background|noise)[^\]]*\]"
    r"|\*[^*]{2,40}\*", re.I)


def hindi_share(turns):
    """Fraction of word tokens that are recognisable romanized Hindi."""
    words = re.findall(r"[a-z]+", " ".join(t["text"] for t in turns).lower())
    if not words:
        return 0.0
    return sum(1 for w in words if w in HINDI_MARKERS) / len(words)


def hindi_markers_used(turns):
    words = re.findall(r"[a-z]+", " ".join(t["text"] for t in turns).lower())
    return len({w for w in words if w in HINDI_MARKERS})


def devanagari_share(turns):
    text = " ".join(t["text"] for t in turns)
    dev = len(_DEVANAGARI.findall(text))
    return dev / max(dev + len(_LATIN_LETTER.findall(text)), 1)


def post_validate(cls, turns):
    """Problems worth a retry. Empty list means the conversation is acceptable."""
    problems = []
    # The corpus is romanized Hinglish; asking for it in the prompt is not enough, so a
    # Devanagari answer is regenerated rather than kept.
    foreign = non_latin_chars(turns)
    if foreign > MAX_NON_LATIN_CHARS:
        problems.append(f"{foreign} non-Latin character(s); corpus must be romanized Latin")

    used = hindi_markers_used(turns)
    if used < MIN_HINDI_MARKERS:
        problems.append(f"only {used} distinct Hindi word(s); must be Hindi-English code-mixed")

    for t in turns:
        tx = t["text"]
        # Masked digits defeat the point: the corpus feeds a regex/NER redaction stage, so
        # identifiers must look like real numbers.
        if _MASKED_DIGITS.search(tx):
            problems.append(f"turn {t['turn_id']} has masked placeholder digits, needs real-looking ones")
        if _NARRATION.search(tx):
            problems.append(f"turn {t['turn_id']} contains stage direction / narration")
        stripped = tx.rstrip()
        # A turn ending in a dash is interrupted speech ("aap galat number laga—"), which
        # is realistic on a real call; only a bare word ending is actual truncation.
        if stripped and stripped[-1] not in ".!?\"'’”)…—–-," and not stripped[-1].isdigit():
            problems.append(f"turn {t['turn_id']} looks cut off mid-sentence")

    for a, b in zip(turns, turns[1:]):
        if a["speaker"] == b["speaker"]:
            problems.append(f"turn {b['turn_id']} repeats the previous speaker")
    if len(turns) < 6:
        problems.append(f"only {len(turns)} turns; minimum is 6")
    tagged = {g for t in turns for g in t["tactics"]}
    if cls == "legitimate_authority_hard_negative":
        # Regenerate rather than keep a "legitimate" call that actually behaves like a scam.
        problems.extend(validate_hard_negative({"turns": turns})[0])
    elif cls in SCAM_CLASSES:
        if not tagged:
            problems.append("scam call has no tactic tags at all")
    if not any(t["speaker"] == "caller" for t in turns) or not any(t["speaker"] == "victim" for t in turns):
        problems.append("conversation is missing one of the two speakers")
    return problems


class BlockedError(RuntimeError):
    """The provider returned no usable text (safety filter, truncation, empty candidate)."""


def _completion(prompt):
    """Return the model's raw response text as a JSON string."""
    if PROVIDER == "vertex":
        return _completion_vertex(prompt)
    return _completion_groq(prompt)


def _completion_vertex(prompt):
    from google.genai import types

    cfg = dict(
        temperature=TEMPERATURE,
        max_output_tokens=MAX_COMPLETION_TOKENS,
        response_mime_type="application/json",
        # These transcripts depict fraud for a scam-detection dataset, so the default
        # filters flag them; only high-severity content is still blocked.
        safety_settings=[
            types.SafetySetting(category=c, threshold="BLOCK_ONLY_HIGH")
            for c in ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                      "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")
        ],
    )
    # gemini-2.5-* "think" by default; those tokens eat max_output_tokens and can starve
    # the answer. These conversations need no deliberation.
    if _fast_mode.is_set():
        try:
            resp = _client.models.generate_content(
                model=MODEL_NAME, contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0), **cfg))
            return _vertex_text(resp)
        except Exception as e:
            if "thinking" in str(e).lower():
                _fast_mode.clear()
                print("    note: thinking_budget unsupported, continuing without it")
            else:
                raise
    resp = _client.models.generate_content(
        model=MODEL_NAME, contents=prompt, config=types.GenerateContentConfig(**cfg))
    return _vertex_text(resp)


def _vertex_text(resp):
    text = getattr(resp, "text", None)
    if text:
        return text
    reason = "empty response"
    cand = (getattr(resp, "candidates", None) or [None])[0]
    if cand is not None and getattr(cand, "finish_reason", None):
        reason = f"finish_reason={cand.finish_reason}"
    if getattr(resp, "prompt_feedback", None):
        reason += f" prompt_feedback={resp.prompt_feedback}"
    raise BlockedError(reason)


def _completion_groq(prompt):
    kwargs = dict(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=TEMPERATURE,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )
    if _fast_mode.is_set():
        try:
            return _client.chat.completions.create(
                reasoning_effort="low", **kwargs).choices[0].message.content
        except Exception as e:
            if "reasoning_effort" in str(e).lower() or isinstance(e, TypeError):
                _fast_mode.clear()
                print("    note: reasoning_effort unsupported, continuing without it")
            else:
                raise
    return _client.chat.completions.create(**kwargs).choices[0].message.content


_RETRY_HINT = re.compile(r"try again in ([\d.]+)\s*(ms|s)\b")


def _retry_delay(msg, attempt):
    """Honour the server's own retry hint when it gives one.

    Groq's 429 replies say e.g. "Please try again in 637.5ms"; sleeping the usual
    exponential 4-32s instead would waste most of the per-minute token budget.
    """
    hint = _RETRY_HINT.search(msg)
    if hint:
        secs = float(hint.group(1)) / (1000 if hint.group(2) == "ms" else 1)
        return min(60.0, secs + 0.25 + 0.5 * random.random())
    return min(60.0, 4 * (2 ** attempt)) * (0.7 + 0.6 * random.random())


def generate_conversation(spec):
    prompt = build_prompt(spec)
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            parsed = json.loads(_completion(prompt))
            turns = normalize_turns(spec["class"], parsed["turns"])
            problems = post_validate(spec["class"], turns)
            if problems:
                if attempt < MAX_RETRIES - 1:
                    last_error = ValueError("; ".join(problems))
                    continue
                # Out of retries. Keep the conversation rather than lose it, but say so
                # loudly: silently storing a record that fails validation is how bad rows
                # end up in the training set.
                print(f"    ACCEPTED WITH PROBLEMS {spec['conversation_id']}: "
                      f"{'; '.join(problems)}", flush=True)
            return {
                "conversation_id": spec["conversation_id"],
                "pair_id": spec["pair_id"],
                "class": spec["class"],
                "is_vishing": spec["class"] in SCAM_CLASSES,
                "impersonated_org": spec["org"],
                "pretext": spec["pretext"],
                "victim_persona": spec["persona"],
                "turns": turns,
            }
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            transient = ("429" in msg or "503" in msg or "502" in msg or "rate" in msg
                         or "timeout" in msg or "connection" in msg
                         or "resource_exhausted" in msg or "quota" in msg
                         or "unavailable" in msg or "deadline" in msg
                         or isinstance(e, (json.JSONDecodeError, KeyError, ValueError,
                                           BlockedError)))
            if transient and attempt < MAX_RETRIES - 1:
                time.sleep(_retry_delay(msg, attempt))
            elif attempt < MAX_RETRIES - 1:
                time.sleep(2)
            else:
                break

    raise RuntimeError(f"{spec['conversation_id']} failed after {MAX_RETRIES} attempts: {last_error}")


def _scenario_commons():
    return {
        "city": random.choice(CITIES),
        "victim_name": random.choice(VICTIM_NAMES),
        "persona": random.choice(VICTIM_PERSONAS),
        "codeswitch": random.choice(CODESWITCH_RATIOS),
    }


def _spec(conv_id, pair_id, cls, org, pretext, commons, n_turns, flavour=None, subtype_key=None):
    if cls in SCAM_CLASSES:
        outcome = random.choice(SCAM_OUTCOMES)
    elif cls == "legitimate_authority_hard_negative":
        outcome = random.choice(LEGIT_OUTCOMES)
    else:
        outcome = random.choice(BENIGN_TEXTURES)

    spec = {
        "conversation_id": conv_id,
        "pair_id": pair_id,
        "class": cls,
        "org": org,
        "pretext": pretext,
        "n_turns": n_turns,
        "flavour": flavour,
        "caller_name": random.choice(CALLER_NAMES),
        "time_of_day": random.choice(TIMES_OF_DAY),
        "manner": random.choice(CALLER_MANNERS),
        "opening": random.choice(OPENING_STYLES),
        "outcome": outcome,
        "ref_id": None if cls == "benign" else _fake_ref(subtype_key or "law"),
        **commons,
    }
    if cls in SCAM_CLASSES:
        spec["amount"] = _amount_for(cls, subtype_key)
        spec["upi"] = _fake_upi()
    return spec


def build_plan(n_total, start_conv_num, start_pair_num):
    """Roughly 23% contrastive pairs / 38.5% other_scam / 38.5% benign, which comes out
    balanced 50/50 between vishing and non-vishing."""
    n_pairs = round(n_total * 0.233 / 2)
    n_other = round(n_total * 0.3835)
    n_benign = n_total - (n_pairs * 2) - n_other
    if n_benign < 0:
        n_other += n_benign
        n_benign = 0

    specs = []
    conv_num = start_conv_num
    pair_num = start_pair_num

    for _ in range(n_pairs):
        pair_id = f"pair_{pair_num:03d}"
        pair_num += 1
        commons = _scenario_commons()
        org = random.choice(DA_ORGS)
        pretext = random.choice(DA_PRETEXTS)
        n_turns = random.choice([7, 8, 9, 10, 11, 12, 13, 15])
        for cls in ("digital_arrest_scam", "legitimate_authority_hard_negative"):
            specs.append(_spec(f"syn_{conv_num:03d}", pair_id, cls, org, pretext, commons, n_turns))
            conv_num += 1

    subtype_keys = list(OTHER_SCAM_SUBTYPES)
    for i in range(n_other):
        key = subtype_keys[i % len(subtype_keys)]
        info = OTHER_SCAM_SUBTYPES[key]
        specs.append(_spec(f"syn_{conv_num:03d}", None, "other_scam", info["org"], info["pretext"],
                           _scenario_commons(), random.choice([6, 7, 8, 9, 10, 11, 12]),
                           flavour=info["flavour"], subtype_key=key))
        conv_num += 1

    for i in range(n_benign):
        call_type = BENIGN_CALL_TYPES[i % len(BENIGN_CALL_TYPES)]
        specs.append(_spec(f"syn_{conv_num:03d}", None, "benign", call_type["org"],
                           call_type["pretext"], _scenario_commons(),
                           random.choice([6, 7, 8, 9, 10, 11])))
        conv_num += 1

    random.shuffle(specs)
    return specs


def load_existing():
    if not os.path.exists(OUTPUT_FILE):
        return [], 0, 1
    records = []
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    max_conv, max_pair = -1, 0
    for r in records:
        cid = str(r.get("conversation_id", ""))
        if cid.startswith("syn_"):
            try:
                max_conv = max(max_conv, int(cid.split("_")[1]))
            except ValueError:
                pass
        pid = r.get("pair_id")
        if pid and str(pid).startswith("pair_"):
            try:
                max_pair = max(max_pair, int(str(pid).split("_")[1]))
            except ValueError:
                pass
    return records, max_conv + 1, max_pair + 1


def orphan_pair_specs(records, start_conv_num):
    """One spec per half-finished contrastive pair, to complete it."""
    members = defaultdict(list)
    for r in records:
        if r.get("pair_id"):
            members[r["pair_id"]].append(r)

    specs = []
    conv_num = start_conv_num
    for pair_id, group in sorted(members.items()):
        if len(group) != 1:
            continue
        have = group[0]
        missing = ("legitimate_authority_hard_negative"
                   if have["class"] == "digital_arrest_scam" else "digital_arrest_scam")
        commons = {
            "city": random.choice(CITIES),
            "victim_name": random.choice(VICTIM_NAMES),
            "persona": have["victim_persona"],
            "codeswitch": random.choice(CODESWITCH_RATIOS),
        }
        specs.append(_spec(f"syn_{conv_num:03d}", pair_id, missing, have["impersonated_org"],
                           have["pretext"], commons, len(have["turns"])))
        conv_num += 1
    return specs


def infer_subtype(rec):
    """Best-effort map an existing other_scam record back to its subtype key."""
    hay = f"{rec.get('pretext', '')} {rec.get('impersonated_org', '')}".lower()
    for key, info in OTHER_SCAM_SUBTYPES.items():
        if info["pretext"].lower() in hay:
            return key
    keywords = {
        "lottery": ("lottery", "kbc", "lucky draw", "prize"),
        "job": ("job", "hr ", "consultancy", "placement", "recruit"),
        "loan": ("loan", "fintech", "sanction"),
        "courier": ("courier", "parcel", "fedex", "dhl", "bluedart", "customs"),
        "otp_bank": ("fraud desk", "card", "otp", "transaction"),
        "tech_support": ("microsoft", "amazon", "tech support", "virus", "compromis"),
        "investment": ("invest", "trading", "crypto", "stock", "advisory"),
        "electricity": ("electric", "bill overdue", "disconnect"),
        "sextortion": ("blackmail", "intimate", "video"),
        "matrimonial": ("matrimon", "profile verification"),
        "army_fake": ("army", "marketplace"),
    }
    for key, words in keywords.items():
        if any(w in hay for w in words):
            return key
    return random.choice(list(OTHER_SCAM_SUBTYPES))


def regeneration_specs(records, ids):
    """Specs that rewrite existing records in place, keeping their identity and scenario.

    conversation_id, pair_id and class are preserved so contrastive pairs and the class
    balance survive. Legacy benign records (org "none", pretext "casual conversation")
    predate the current benign call types, so they are given one.
    """
    wanted = set(ids)
    specs = []
    for r in records:
        if r["conversation_id"] not in wanted:
            continue
        cls = r["class"]
        org, pretext = r.get("impersonated_org"), r.get("pretext")
        subtype = None
        if cls == "benign" and (not org or org.lower() in ("none", "n/a")):
            pick = random.choice(BENIGN_CALL_TYPES)
            org, pretext = pick["org"], pick["pretext"]
        if cls == "other_scam":
            subtype = infer_subtype(r)
            org = OTHER_SCAM_SUBTYPES[subtype]["org"]
            pretext = OTHER_SCAM_SUBTYPES[subtype]["pretext"]
        commons = {
            "city": random.choice(CITIES),
            "victim_name": random.choice(VICTIM_NAMES),
            "persona": r.get("victim_persona") or random.choice(VICTIM_PERSONAS),
            "codeswitch": random.choice(CODESWITCH_RATIOS),
        }
        specs.append(_spec(r["conversation_id"], r.get("pair_id"), cls, org, pretext, commons,
                           random.choice([7, 8, 9, 10, 11, 12, 13]),
                           flavour=OTHER_SCAM_SUBTYPES[subtype]["flavour"] if subtype else None,
                           subtype_key=subtype))
    return specs


def replace_records(new_records):
    """Swap regenerated conversations into the file in place, atomically."""
    by_id = {r["conversation_id"]: r for r in new_records}
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    merged = [by_id.get(r["conversation_id"], r) for r in recs]
    tmp = OUTPUT_FILE + ".replacing"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, OUTPUT_FILE)
    return sum(1 for r in recs if r["conversation_id"] in by_id)


def select_ids(records, selector):
    """Resolve a --regenerate selector into conversation ids."""
    ids = set()
    for part in (p.strip() for p in selector.split(",") if p.strip()):
        if part == "untagged":
            ids |= {r["conversation_id"] for r in records
                    if any("tactics" not in t for t in r["turns"])}
        elif part in ("nonlatin", "devanagari"):
            ids |= {r["conversation_id"] for r in records
                    if non_latin_chars(r["turns"]) > MAX_NON_LATIN_CHARS}
        elif part == "invalid":
            # Anything the current validation rules reject, so tightening a rule and
            # re-running is enough to repair the corpus.
            ids |= {r["conversation_id"] for r in records
                    if post_validate(r["class"], r["turns"])}
        elif part.startswith("ids:"):
            ids |= {x.strip() for x in part[4:].split("|") if x.strip()}
        else:
            raise SystemExit(f"unknown --regenerate selector {part!r} "
                             "(use untagged, devanagari, or ids:syn_001|syn_002)")
    return sorted(ids)


def run_batch(specs, workers, label, append=True):
    done, failed = [], []
    lock = threading.Lock()
    total = len(specs)

    out_f = open(OUTPUT_FILE, "a", encoding="utf-8") if append else None
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for i, spec in enumerate(specs):
                if i < workers:
                    time.sleep(0.15)  # stagger only the opening burst
                futures[pool.submit(generate_conversation, spec)] = spec

            for fut in as_completed(futures):
                spec = futures[fut]
                try:
                    conv = fut.result()
                except Exception as e:
                    with lock:
                        failed.append(spec["conversation_id"])
                        print(f"[{len(done) + len(failed)}/{total}] {label} FAILED "
                              f"{spec['conversation_id']} ({spec['class']}): {e}", flush=True)
                    continue
                with lock:
                    if out_f is not None:
                        out_f.write(json.dumps(conv, ensure_ascii=False) + "\n")
                        out_f.flush()
                    done.append(conv)
                    n = len(done) + len(failed)
                    if n % 10 == 0 or n == total:
                        print(f"[{n}/{total}] {label} ok={len(done)} failed={len(failed)}", flush=True)
    finally:
        if out_f is not None:
            out_f.close()
    return done, failed


_SENSITIVE = re.compile(r"\b(otp|o\.t\.p\.?|cvv|pin|password|one[-\s]time password)\b")
# A legitimate caller is *expected* to say "we never ask for your OTP", so a bare keyword
# match cannot distinguish a disclaimer from a solicitation. Look at the surrounding clause.
# Enumerating every way a caller can promise *not* to ask for a credential is hopeless
# ("never ask", "never need", "never solicit", "maangenge nahi", "maang nahi karte",
# "nahi chahiye", "नहीं मांगते"...). Negation markers are a small closed set, so detect
# those instead and treat any negated clause as a disclaimer.
_NEGATION = re.compile(
    r"(\bnot\b|n'?t\b|\bnever\b|\bno\b|\bnone\b|\bneither\b|\bnor\b|\bwithout\b|"
    r"\bnahi+n?\b|\bnai\b|\bmat\b|\bbina\b|\bbilkul nahi|"
    r"नहीं|नही|\bमत\b|बिना)")
_SOLICIT = re.compile(
    r"(share|tell me|give me|read out|read me|send me|provide|forward|repeat|confirm the|"
    r"enter the|batao|bata d|bata de|bhej|likh|type)")
_SECRECY = re.compile(
    r"(don'?t tell anyone|do not tell anyone|tell no ?one|tell nobody|don'?t inform|"
    r"do not inform|keep this (confidential|secret|between)|kisi ko mat bat|"
    r"kisi ko nahi bat|family ko mat|don'?t discuss this with)")
_HOLD = re.compile(r"(stay on the line|don'?t disconnect|do not disconnect|don'?t hang up|"
                   r"line pe rahi|call mat kat)")


def _sentences(text):
    return [s for s in re.split(r"[.!?।\n]+", text) if s.strip()]


def validate_hard_negative(conv):
    """Returns (violations, warnings). Violations are disqualifying for a hard negative."""
    violations, warnings = [], []
    bad_tactics = {"fear", "threat", "isolation", "financial_coercion"}

    for t in conv["turns"]:
        leaked = bad_tactics & set(t.get("tactics", []))
        if leaked:
            violations.append(f"turn {t['turn_id']} tagged {sorted(leaked)}")
        if t["speaker"] != "caller":
            continue
        low = t["text"].lower()
        for sent in _sentences(low):
            if not _SENSITIVE.search(sent):
                continue
            negated = bool(_NEGATION.search(sent))
            asked = bool(_SOLICIT.search(sent))
            if negated and not asked:
                continue  # "we never ask for your OTP" - exactly what a real org says
            if asked and not negated:
                violations.append(f"turn {t['turn_id']} solicits a credential: "
                                  f"...{sent.strip()[:70]}")
            elif asked and negated:
                # e.g. "don't worry, just share the OTP" - genuinely needs human eyes
                warnings.append(f"turn {t['turn_id']} negated solicitation, check manually: "
                                f"...{sent.strip()[:70]}")
        if _SECRECY.search(low):
            violations.append(f"turn {t['turn_id']} demands secrecy")
        if _HOLD.search(low):
            warnings.append(f"turn {t['turn_id']} uses hold/stay-on-line phrasing")
    return violations, warnings


def validate_benign(conv):
    return [f"turn {t['turn_id']} tactics {t['tactics']}"
            for t in conv["turns"] if t.get("tactics")]


def sort_file_by_id():
    """Rewrite the file in conversation_id order (concurrent writes land out of order)."""
    try:
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            recs = [json.loads(l) for l in f if l.strip()]
        recs.sort(key=lambda r: int(str(r["conversation_id"]).split("_")[1]))
        tmp = OUTPUT_FILE + ".sorting"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, OUTPUT_FILE)
        return len(recs)
    except Exception as e:
        print(f"  (skipped final sort: {e})")
        return None


def report(new_records):
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    if new_records:
        print("New this run:", dict(Counter(c["class"] for c in new_records)))
        print("Vishing split:", dict(Counter("vishing" if c["is_vishing"] else "non_vishing"
                                             for c in new_records)))
        turn_counts = [len(c["turns"]) for c in new_records]
        print(f"Turns per conversation: min={min(turn_counts)} max={max(turn_counts)} "
              f"avg={sum(turn_counts) / len(turn_counts):.1f}")

        hard_negs = [c for c in new_records if c["class"] == "legitimate_authority_hard_negative"]
        checked = [(c["conversation_id"], *validate_hard_negative(c)) for c in hard_negs]
        bad = [(cid, v) for cid, v, _ in checked if v]
        warned = [(cid, w) for cid, _, w in checked if w]
        print(f"\nHard negatives clean: {len(hard_negs) - len(bad)}/{len(hard_negs)}")
        for cid, v in bad[:15]:
            print(f"  VIOLATION {cid}: {v}")
        print(f"Hard negatives with warnings (ambiguous, worth eyeballing): {len(warned)}")
        for cid, w in warned[:5]:
            print(f"  warn {cid}: {w}")

        benigns = [c for c in new_records if c["class"] == "benign"]
        bad_b = [(c["conversation_id"], v) for c in benigns if (v := validate_benign(c))]
        print(f"Benign calls clean: {len(benigns) - len(bad_b)}/{len(benigns)}")
        for cid, v in bad_b[:15]:
            print(f"  {cid}: {v}")

    with open(OUTPUT_FILE, encoding="utf-8") as f:
        allrecs = [json.loads(l) for l in f if l.strip()]

    print(f"\nWhole file: {len(allrecs)} conversations")
    print("Class mix:", dict(Counter(r["class"] for r in allrecs)))
    ids = [r["conversation_id"] for r in allrecs]
    print("Duplicate ids:", len(ids) - len(set(ids)))

    members = Counter(r["pair_id"] for r in allrecs if r.get("pair_id"))
    incomplete = [p for p, n in members.items() if n != 2]
    print(f"Contrastive pairs: {len(members)} total, {len(incomplete)} incomplete "
          f"{sorted(incomplete)[:10] if incomplete else ''}")

    texts = [" ".join(t["text"] for t in r["turns"]).strip().lower() for r in allrecs]
    print("Exact duplicate conversations:", len(texts) - len(set(texts)))
    openings = Counter(" ".join(r["turns"][0]["text"].split()[:8]).lower()
                       for r in allrecs if r["turns"])
    repeated = [(o, n) for o, n in openings.most_common(5) if n > 1]
    print("Most repeated opening phrasings:" if repeated else "No repeated opening phrasings.")
    for o, n in repeated:
        print(f"  {n}x  {o}...")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--count", type=int, help="how many new conversations to generate")
    g.add_argument("--target", type=int, help="generate until the file holds this many in total")
    g.add_argument("--regenerate", metavar="SELECTOR",
                   help="rewrite existing conversations in place, keeping their id, pair and "
                        "class. Selectors, comma-separated: 'untagged' (no per-turn tactics), "
                        "'nonlatin' (stray non-Latin script), 'ids:syn_001|syn_002'")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="concurrent API calls")
    ap.add_argument("--provider", choices=sorted(DEFAULT_MODELS), default="vertex",
                    help="which API to generate with (default: vertex)")
    ap.add_argument("--model", default=None,
                    help=f"model override; defaults per provider: {DEFAULT_MODELS}")
    ap.add_argument("--no-repair-pairs", action="store_true",
                    help="skip completing contrastive pairs left half-finished earlier")
    ap.add_argument("--seed", type=int, default=None, help="seed the scenario sampling")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    init_provider(args.provider, args.model)

    existing, next_conv, next_pair = load_existing()

    if args.regenerate:
        ids = select_ids(existing, args.regenerate)
        if not ids:
            print("Nothing matched that selector; file unchanged.")
            return
        specs = regeneration_specs(existing, ids)
        print(f"Regenerating {len(specs)} existing conversation(s) in place with "
              f"{args.workers} workers ({PROVIDER}:{MODEL_NAME}).", flush=True)
        started = time.time()
        done, failed = run_batch(specs, args.workers, "regen", append=False)
        if done:
            swapped = replace_records(done)
            print(f"\nReplaced {swapped} conversation(s) in "
                  f"{(time.time() - started) / 60:.1f} min", flush=True)
        if failed:
            print(f"{len(failed)} could not be regenerated and were left untouched: "
                  f"{failed[:10]}", flush=True)
        report(done)
        return

    count = args.count if args.count is not None else max(0, args.target - len(existing))
    print(f"File holds {len(existing)} conversations. New ids start at syn_{next_conv:03d}, "
          f"pair_{next_pair:03d}. Generating {count} with {args.workers} workers "
          f"({PROVIDER}:{MODEL_NAME}).", flush=True)

    all_new = []
    started = time.time()

    if not args.no_repair_pairs:
        repairs = orphan_pair_specs(existing, next_conv)
        if repairs:
            print(f"\nCompleting {len(repairs)} half-finished contrastive pair(s) first...", flush=True)
            done, failed = run_batch(repairs, args.workers, "repair")
            all_new.extend(done)
            next_conv += len(repairs)
            print(f"Pair repair: {len(done)} ok, {len(failed)} failed", flush=True)

    if count > 0:
        specs = build_plan(count, next_conv, next_pair)
        print(f"\nGenerating {len(specs)} conversations...", flush=True)
        done, failed = run_batch(specs, args.workers, "main")
        all_new.extend(done)
        if failed:
            print(f"\n{len(failed)} conversation(s) failed; re-run to top up the shortfall.",
                  flush=True)

    mins = max((time.time() - started) / 60, 1e-6)
    print(f"\nAppended {len(all_new)} conversations in {mins:.1f} min "
          f"({len(all_new) / mins:.1f}/min)", flush=True)

    sort_file_by_id()
    report(all_new)


if __name__ == "__main__":
    main()
