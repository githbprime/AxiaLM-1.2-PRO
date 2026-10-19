# %% 0 - setup and config
# !pip install -q datasets            (Colab already has it)
# !pip install -q deep-translator     (optional: lets you try a real translator in tests, see TRANSLATE_HOOK)
import os, re, sys, json, math, time, random, heapq, unicodedata, zlib, itertools, glob, shutil
import datetime as dt
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from decimal import Decimal, InvalidOperation, getcontext, ROUND_FLOOR
import numpy as np

USE_DRIVE = False  # True (Colab): everything lives on your Google Drive. False: runtime files + automatic .pt downloads/pick-up
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    WORK = "/content/drive/MyDrive/axialm_work"
elif os.path.isdir("/kaggle/working"):
    WORK = "/kaggle/working"
elif os.path.isdir("/content"):
    WORK = "/content/axialm_work"
else:
    WORK = "./axialm_work"
os.makedirs(WORK, exist_ok=True)

VERSION = "v1"  # bump when the tokenizer / data generators change, so old caches are ignored
CKPT_PATH = f"{WORK}/axialm_1_2_pro_{VERSION}.pt"
TOK_PATH = f"{WORK}/axialm_tokenizer_{VERSION}.json"
DB_CACHE = f"{WORK}/axialm_wiki_facts_{VERSION}.tsv"      # raw wikipedia facts (cached)
DB_EXPORT = f"{WORK}/axialm_facts_{VERSION}.tsv"          # final fact database: title <tab> keywords <tab> fact

RUN = "full"  # "full" = the plan (6.5B tokens) | "colab" = 1B tokens | "smoke" = integration test
RUNS = {
    #            training tokens   (web, chat, synth) docs used to train the tokenizer
    "smoke": dict(tokens=6_000_000, tok_docs=(1500, 1500, 6000), eval_every=100, probe_every=200, save_every=100),
    "colab": dict(tokens=1_000_000_000, tok_docs=(20000, 20000, 40000), eval_every=1000, probe_every=5000, save_every=1000),
    "full": dict(tokens=6_500_000_000, tok_docs=(20000, 20000, 40000), eval_every=2000, probe_every=10000, save_every=1000),
}
MODEL = "pro"  # the 6.5M-parameter model from the plan (6,536,640 params)

# plan blend: 50 / 30 / 20 (web / chat / procedural) rescaled so the agent workflow gets 15% (section 5)
MIX = {"web": 0.43, "chat": 0.25, "proc": 0.17, "agent": 0.15}
USE_WEB = True        # stream FineWeb-Edu (falls back to the synthetic buckets if unreachable)
USE_CHAT_DATA = True  # stream UltraChat
USE_WIKI_DB = True    # add ~2000 simple-wikipedia definitions to the fact database (the "database tool")
WIKI_DB_ENTRIES = 2000

SEED = 1234
BATCH = 64
LR = 1.5e-3
MIN_LR_FRAC = 0.10
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
QAT_FRAC = 0.30            # last 30% of training runs with 8-level (3-bit) fake quantization
SPECIAL_WEIGHT = 2.0       # extra loss weight on the control tokens (</think>, <call>, </call>, <eos> ...)
NUM_WORKERS = 2            # CPU processes that stream + tokenize + generate data
MAX_HOURS = 10.5           # stop cleanly (checkpoint saved + downloaded) before Colab's session limit; re-run to resume
DOWNLOAD_EVERY_HOURS = 0.1 # USE_DRIVE = False on Colab: download the .pt this often, so a disconnect loses at most this much
COMPILE = True             # torch.compile (falls back automatically if it fails)
RESUME = True

THINK_LIMIT = 60           # a think block longer than this gets </think> forced (the goboscript engine does the same)
CALL_LIMIT = 24            # same for a tool call
RESULT_LIMIT = 80          # tool results are cut to this many tokens
MAX_CALLS = 3              # tool calls per reply

getcontext().prec = 40


@dataclass
class Config:
    vocab_size: int = 4000
    d_model: int = 192
    n_layers: int = 14
    n_heads: int = 6
    n_kv_heads: int = 2
    ffn_dim: int = 544
    ctx: int = 512
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    quant_levels: int = 8      # 8 levels = 3 bits; one character (69 states) holds 2 weights because 8*8 = 64 <= 69
    quant_group: int = 32      # one float scale per 32 weights along the input dimension

    @property
    def head_dim(self):
        return self.d_model // self.n_heads


PRESETS = {"pro": dict()}   # d192 / 14 layers / ffn 544 -> 6,536,640 params


def count_params(c):
    kv = c.n_kv_heads * c.head_dim
    per_layer = 2 * c.d_model * c.d_model + 2 * c.d_model * kv + 3 * c.d_model * c.ffn_dim + 2 * c.d_model
    return c.vocab_size * c.d_model + c.n_layers * per_layer + c.d_model


# Weights are stored as characters of the 69-character alphabet (printable ASCII without uppercase). With L levels per weight,
# k weights share m characters when L**k <= 69**m:  8 levels -> 2 weights per character, 16 -> 3 per 2, 69 -> 1 per 1.
PACK_ALPHABET = "".join(chr(c) for c in range(32, 127) if not chr(c).isupper())
assert len(PACK_ALPHABET) == 69


def char_packing(levels):
    best = None
    for k in range(1, 9):
        for m in range(1, 7):
            if levels ** k <= 69 ** m and (best is None or m / k < best[0] - 1e-9):
                best = (m / k, k, m)
    return best[1], best[2]


def pack_codes(codes, levels):
    """codes (each 0..levels-1) -> string. k codes -> integer sum(code_j * levels**j) -> m base-69 characters, little-endian."""
    k, m = char_packing(levels)
    codes = list(codes) + [0] * (-len(codes) % k)
    out = []
    for i in range(0, len(codes), k):
        v = sum(codes[i + j] * levels ** j for j in range(k))
        for _ in range(m):
            v, d = divmod(v, 69)
            out.append(PACK_ALPHABET[d])
    return "".join(out)


def unpack_codes(text, levels, n):
    k, m = char_packing(levels)
    codes = []
    for i in range(0, len(text), m):
        v = sum(PACK_ALPHABET.index(text[i + j]) * 69 ** j for j in range(m))
        for _ in range(k):
            v, d = divmod(v, levels)
            codes.append(d)
    return codes[:n]


def size_report(c):
    P = count_params(c)
    norms = (2 * c.n_layers + 1) * c.d_model
    w = P - norms
    k, m = char_packing(c.quant_levels)
    chars = math.ceil(w * m / k) + 3 * (w // c.quant_group) + 6 * norms   # codes + 16-bit scales (3 chars) + fp32 norms (6 chars)
    kv = c.n_layers * c.ctx * 2 * c.n_kv_heads * c.head_dim
    print(f"params {P:,} | {c.quant_levels} levels, {k} weight(s) per {m} character(s) -> est. packed weights {chars / 1e6:.2f} MB "
          f"(Scratch project.json cap is 5 MB; vocab/merges/fact db/code come on top) | KV cache at full context {kv:,} values")
    if chars > 4.4e6:
        print("  !! close to / over the 5 MB project.json budget - fewer layers, smaller ffn_dim or fewer levels would help.")


CFG = Config(**PRESETS[MODEL])
RUNCFG = RUNS[RUN]
print(f"model={MODEL} run={RUN} work dir={WORK}")
size_report(CFG)

# %% 1 - text normalization and tokenizer (lowercase BPE; same pre-tokenizer the Scratch engine mirrors)
SPECIALS = ["<pad>", "<bos>", "<|user|>", "<|assistant|>", "<eos>", "<think>", "</think>",
            "<call>", "</call>", "<result>", "</result>"]
PAD, BOS, USER, ASSIST, END, THINK, THINK_END, CALL, CALL_END, RESULT, RESULT_END = range(11)
N_SPECIAL = len(SPECIALS)
BASE_CHARS = ["\n"] + [chr(c) for c in range(32, 127) if not chr(c).isupper()]

_FOLD = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " "}
_BAD = re.compile(r"[^\n -~]")
_TRAIL = re.compile(r" +\n")
_BLANKS = re.compile(r"\n{3,}")
PRETOK = re.compile(r"\n+| ?[a-z]+|[0-9]| ?[^a-z0-9\s]+| +")


def norm(text):
    if not text.isascii():
        for k, v in _FOLD.items():
            text = text.replace(k, v)
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ").lower()
    text = _BAD.sub("", text)
    text = _TRAIL.sub("\n", text)
    text = _BLANKS.sub("\n\n", text)
    return text.strip()


def train_bpe(counts, vocab_size):
    syms = list(BASE_CHARS)
    str2id = {s: i for i, s in enumerate(syms)}
    words, freqs = [], []
    for w, c in counts.items():
        words.append([str2id[ch] for ch in w])
        freqs.append(c)
    pair_count, where = defaultdict(int), defaultdict(set)
    for wi, w in enumerate(words):
        for pr in zip(w, w[1:]):
            pair_count[pr] += freqs[wi]
            where[pr].add(wi)
    heap = [(-c, pr) for pr, c in pair_count.items()]
    heapq.heapify(heap)
    merges, target = [], vocab_size - N_SPECIAL
    while len(syms) < target and heap:
        negc, pr = heapq.heappop(heap)
        cur = pair_count.get(pr, 0)
        if cur <= 0:
            continue
        if cur != -negc:
            heapq.heappush(heap, (-cur, pr))
            continue
        a, b = pr
        s = syms[a] + syms[b]
        if s in str2id:
            new = str2id[s]
        else:
            new = len(syms)
            syms.append(s)
            str2id[s] = new
        merges.append(pr)
        touched = set()
        for wi in where.pop(pr):
            w = words[wi]
            nw, i, n, changed = [], 0, len(w), False
            while i < n:
                if i < n - 1 and w[i] == a and w[i + 1] == b:
                    nw.append(new)
                    i += 2
                    changed = True
                else:
                    nw.append(w[i])
                    i += 1
            if not changed:
                continue
            f = freqs[wi]
            for p in zip(w, w[1:]):
                pair_count[p] -= f
            for p in zip(nw, nw[1:]):
                pair_count[p] += f
                where[p].add(wi)
                touched.add(p)
            words[wi] = nw
        pair_count.pop(pr, None)
        for p in touched:
            if pair_count[p] > 0:
                heapq.heappush(heap, (-pair_count[p], p))
    return SPECIALS + syms, [(a + N_SPECIAL, b + N_SPECIAL) for a, b in merges]


class Tokenizer:
    def __init__(self, tokens, merges):
        self.tokens = tokens
        self.tok2id = {t: i for i, t in enumerate(tokens)}
        self.merges = [tuple(m) for m in merges]
        self.rank = {m: i for i, m in enumerate(self.merges)}
        self.merged = {m: self.tok2id[tokens[m[0]] + tokens[m[1]]] for m in self.merges}
        self.cache = {}

    @property
    def vocab_size(self):
        return len(self.tokens)

    @classmethod
    def train(cls, texts, vocab_size):
        counts = Counter()
        for t in texts:
            counts.update(PRETOK.findall(norm(t)))
        print(f"training BPE on {len(counts):,} unique pre-tokens ...")
        return cls(*train_bpe(counts, vocab_size))

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"tokens": self.tokens, "merges": self.merges, "specials": SPECIALS}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["tokens"], d["merges"])

    def _bpe(self, w):
        hit = self.cache.get(w)
        if hit is not None:
            return hit
        ids = [self.tok2id[ch] for ch in w]
        rank = self.rank
        while len(ids) > 1:
            best, best_r = None, 1 << 30
            for i in range(len(ids) - 1):
                r = rank.get((ids[i], ids[i + 1]))
                if r is not None and r < best_r:
                    best_r, best = r, (ids[i], ids[i + 1])
            if best is None:
                break
            new, out, i = self.merged[best], [], 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == best[0] and ids[i + 1] == best[1]:
                    out.append(new)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        if len(self.cache) > 300_000:
            self.cache.clear()
        self.cache[w] = ids
        return ids

    def encode(self, text):
        ids = []
        for w in PRETOK.findall(norm(text)):
            ids.extend(self._bpe(w))
        return ids

    def decode(self, ids):
        return "".join(self.tokens[i] for i in ids if i >= N_SPECIAL)


_JSNUM = re.compile(r"^(?:[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?|0x[0-9a-f]+|0b[01]+|0o[0-7]+)$", re.I)


def scratch_key(tok):
    s = tok.strip()
    if s and _JSNUM.match(s):
        low = s.lower()
        return ("n", float(int(low, 0) if low.startswith(("0x", "0b", "0o")) else float(low)))
    return ("s", tok.lower())


def scratch_collisions(tokens):
    seen, bad = {}, []
    for i, t in enumerate(tokens):
        k = scratch_key(t)
        if k in seen:
            bad.append((tokens[seen[k]], t))
        else:
            seen[k] = i
    return bad

# %% 2 - the tools (this is the reference the goboscript engine copies; the model only ever sees text)
# A tool call is the text between <call> and </call>:   name | arg | arg ...
#   math      | addition|subtraction|multiplication|division|modulo|power|percent  | a | b
#   math      | sqrt | a                      math | solve | a | b | c      (solves  a*x + b = c)
#   datetime  | date   -> year | month | day | weekday      (weekday: 1 = sunday ... 7 = saturday, like scratch)
#   datetime  | time   -> hour | minute | second            (24h clock; goboscript: current_hour, current_minute ...)
#   db        | keywords                     -> a fact sentence, or "none"
#   translate | language | text             -> translated text  (swap tool_translate for the scratch extension block)
# The result is injected as  <result> ... </result>  and the model continues.
SCR_DOW = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
               "november", "december"]
_NUM = re.compile(r"-?\d+(\.\d+)?$")


def _d(s):
    s = s.strip()
    if not _NUM.match(s):
        raise ValueError(s)
    return Decimal(s)


def fmt_dec(d):
    if d == d.to_integral_value():
        return str(int(d))
    q = d.quantize(Decimal("0.0001"))
    if q == 0:
        return "0"
    return format(q, "f").rstrip("0").rstrip(".")


def fmt_commas(s):
    m = re.fullmatch(r"(-?)(\d+)(\.\d+)?", s)
    if not m:
        return s
    sign, whole, frac = m.group(1), m.group(2), m.group(3) or ""
    return sign + (f"{int(whole):,}" if len(whole) > 3 else whole) + frac


def tool_math(args):
    try:
        op, v = args[0].strip(), args[1:]
        if op == "sqrt":
            a = _d(v[0])
            return "error" if a < 0 else fmt_dec(a.sqrt())
        if op == "solve":
            a, b, c = _d(v[0]), _d(v[1]), _d(v[2])
            return "error" if a == 0 else fmt_dec((c - b) / a)
        a, b = _d(v[0]), _d(v[1])
        if op == "addition":
            r = a + b
        elif op == "subtraction":
            r = a - b
        elif op == "multiplication":
            r = a * b
        elif op == "division":
            if b == 0:
                return "error"
            r = a / b
        elif op == "modulo":
            if b == 0:
                return "error"
            r = a - b * (a / b).to_integral_value(rounding=ROUND_FLOOR)   # sign follows the divisor, like scratch
        elif op == "power":
            if b != b.to_integral_value() or not (0 <= b <= 12) or abs(a) > 100000:
                return "error"
            r = a ** int(b)
        elif op == "percent":
            r = a * b / 100
        else:
            return "error"
        return "error" if abs(r) >= Decimal(10) ** 15 else fmt_dec(r)
    except (ValueError, IndexError, InvalidOperation):
        return "error"


def scratch_dow(d):
    return (d.weekday() + 1) % 7 + 1


def tool_datetime(kind, now):
    if kind == "date":
        return f"{now.year} | {now.month} | {now.day} | {scratch_dow(now)}"
    if kind == "time":
        return f"{now.hour} | {now.minute} | {now.second}"
    return "error"


LANGS = ["spanish", "french", "german", "italian", "portuguese"]
_PHRASES = """hello = hola | bonjour | hallo | ciao | ola
good morning = buenos dias | bonjour | guten morgen | buongiorno | bom dia
good afternoon = buenas tardes | bon apres-midi | guten tag | buon pomeriggio | boa tarde
good night = buenas noches | bonne nuit | gute nacht | buonanotte | boa noite
thank you = gracias | merci | danke | grazie | obrigado
please = por favor | s'il vous plait | bitte | per favore | por favor
goodbye = adios | au revoir | auf wiedersehen | arrivederci | tchau
yes = si | oui | ja | si | sim
no = no | non | nein | no | nao
how are you = como estas | comment allez-vous | wie geht es dir | come stai | como vai voce
i love you = te quiero | je t'aime | ich liebe dich | ti amo | eu te amo
where is the library = donde esta la biblioteca | ou est la bibliotheque | wo ist die bibliothek | dov'e la biblioteca | onde fica a biblioteca
where is the bathroom = donde esta el bano | ou sont les toilettes | wo ist die toilette | dov'e il bagno | onde fica o banheiro
i am hungry = tengo hambre | j'ai faim | ich habe hunger | ho fame | estou com fome
what is your name = como te llamas | comment t'appelles-tu | wie heisst du | come ti chiami | qual e o seu nome
i do not understand = no entiendo | je ne comprends pas | ich verstehe nicht | non capisco | eu nao entendo
see you tomorrow = hasta manana | a demain | bis morgen | a domani | ate amanha
good luck = buena suerte | bonne chance | viel gluck | buona fortuna | boa sorte
happy birthday = feliz cumpleanos | joyeux anniversaire | alles gute zum geburtstag | buon compleanno | feliz aniversario
welcome = bienvenido | bienvenue | willkommen | benvenuto | bem-vindo
cat = gato | chat | katze | gatto | gato
dog = perro | chien | hund | cane | cachorro
water = agua | eau | wasser | acqua | agua
i like pizza = me gusta la pizza | j'aime la pizza | ich mag pizza | mi piace la pizza | eu gosto de pizza
the book is on the table = el libro esta sobre la mesa | le livre est sur la table | das buch liegt auf dem tisch | il libro e sul tavolo | o livro esta na mesa
excuse me = perdon | excusez-moi | entschuldigung | scusi | com licenca
i am sorry = lo siento | je suis desole | es tut mir leid | mi dispiace | desculpe
i am happy = estoy feliz | je suis heureux | ich bin glucklich | sono felice | estou feliz
what time is it = que hora es | quelle heure est-il | wie spat ist es | che ora e | que horas sao"""
PHRASEBOOK = {}
for _ln in _PHRASES.split("\n"):
    _en, _rest = _ln.split(" = ")
    PHRASEBOOK[_en.strip()] = dict(zip(LANGS, [x.strip() for x in _rest.split(" | ")]))

TRANSLATE_HOOK = None   # optional: set to fn(language, text) -> text to try a real translator while testing


def tool_translate(lang, text):
    """Placeholder. In Scratch this is replaced by the translate extension block: it is given `text` and `lang`
    and whatever it returns goes back to the model as the <result>."""
    lang, key = lang.strip().lower(), norm(text).rstrip("?.! ")
    if TRANSLATE_HOOK:
        try:
            return norm(TRANSLATE_HOOK(lang, text))
        except Exception:
            return "error"
    if lang in LANGS and key in PHRASEBOOK:
        return PHRASEBOOK[key][lang]
    return "error"


def words_of(s):
    return re.findall(r"[a-z0-9]+", s.lower())


class FactDB:
    """The database tool. Entry = (keywords, fact). A query matches an entry when EVERY query word is one of its
    keywords; among matches the entry with the fewest keywords wins (most specific). No match -> "none"."""

    def __init__(self, entries):
        self.entries = entries
        self.kset = [set(e["k"]) for e in entries]
        self.index = defaultdict(set)
        for i, ks in enumerate(self.kset):
            for w in ks:
                self.index[w].add(i)
        self.train_entries = [e for e in entries if not e["held"]]
        self.held_entries = [e for e in entries if e["held"]]

    def search(self, query):
        ws = set(words_of(query))
        if not ws:
            return "none"
        cand = None
        for w in ws:
            s = self.index.get(w)
            if not s:
                return "none"
            cand = set(s) if cand is None else cand & s
            if not cand:
                return "none"
        return self.entries[min(cand, key=lambda i: (len(self.kset[i]), i))]["f"]


DB = None  # set in the build cell


def run_tool(call_text, now=None, db=None):
    parts = [p.strip() for p in call_text.split("|")]
    name, args = parts[0], parts[1:]
    if name == "math" and args:
        return tool_math(args)
    if name == "datetime" and args:
        return tool_datetime(args[0], now or dt.datetime.now())
    if name == "db" and args:
        return (db or DB).search(" ".join(args))
    if name == "translate" and len(args) >= 2:
        return tool_translate(args[0], " | ".join(args[1:]))
    return "error"

# %% 3 - the fact database (keywords + fact; hand-written core + optional simple-wikipedia definitions)
ALIAS = {"united states": "usa us america", "united kingdom": "uk britain"}
COUNTRY_DATA = """france/paris/europe germany/berlin/europe italy/rome/europe spain/madrid/europe portugal/lisbon/europe
united_kingdom/london/europe ireland/dublin/europe norway/oslo/europe sweden/stockholm/europe finland/helsinki/europe
denmark/copenhagen/europe netherlands/amsterdam/europe belgium/brussels/europe switzerland/bern/europe austria/vienna/europe
poland/warsaw/europe greece/athens/europe ukraine/kyiv/europe japan/tokyo/asia china/beijing/asia india/new_delhi/asia
south_korea/seoul/asia thailand/bangkok/asia vietnam/hanoi/asia indonesia/jakarta/asia philippines/manila/asia
pakistan/islamabad/asia iran/tehran/asia saudi_arabia/riyadh/asia egypt/cairo/africa kenya/nairobi/africa
nigeria/abuja/africa south_africa/pretoria/africa morocco/rabat/africa ethiopia/addis_ababa/africa ghana/accra/africa
algeria/algiers/africa canada/ottawa/north_america united_states/washington_dc/north_america
mexico/mexico_city/north_america cuba/havana/north_america jamaica/kingston/north_america brazil/brasilia/south_america
argentina/buenos_aires/south_america chile/santiago/south_america peru/lima/south_america colombia/bogota/south_america
venezuela/caracas/south_america ecuador/quito/south_america australia/canberra/oceania new_zealand/wellington/oceania"""

PLANET_FACTS = {
    "mercury": "mercury is the smallest planet and the closest one to the sun.",
    "venus": "venus is the hottest planet, even hotter than mercury, because of its thick atmosphere.",
    "earth": "earth is the third planet from the sun and the only planet known to have life.",
    "mars": "mars is the fourth planet from the sun and is called the red planet.",
    "jupiter": "jupiter is the largest planet in the solar system.",
    "saturn": "saturn is the sixth planet from the sun and is famous for its rings.",
    "uranus": "uranus is an ice giant that spins on its side.",
    "neptune": "neptune is the farthest planet from the sun.",
}
ELEMENTS = [("hydrogen", "h", 1), ("helium", "he", 2), ("lithium", "li", 3), ("carbon", "c", 6), ("nitrogen", "n", 7),
            ("oxygen", "o", 8), ("neon", "ne", 10), ("sodium", "na", 11), ("magnesium", "mg", 12), ("aluminum", "al", 13),
            ("silicon", "si", 14), ("sulfur", "s", 16), ("chlorine", "cl", 17), ("potassium", "k", 19),
            ("calcium", "ca", 20), ("iron", "fe", 26), ("copper", "cu", 29), ("zinc", "zn", 30), ("silver", "ag", 47),
            ("tin", "sn", 50), ("gold", "au", 79), ("mercury", "hg", 80), ("lead", "pb", 82)]
ANIMAL_FACTS = {"lion": ("meat", "grasslands"), "tiger": ("meat", "forests"), "elephant": ("plants", "grasslands"),
                "giraffe": ("leaves", "grasslands"), "zebra": ("grass", "grasslands"), "monkey": ("fruit", "forests"),
                "panda": ("bamboo", "forests"), "wolf": ("meat", "forests"), "dolphin": ("fish", "oceans"),
                "eagle": ("small animals", "mountains"), "owl": ("mice", "forests")}
# "title :: fact"   (a leading "p:" marks a person -> "who is ...?" questions)
DEFS = """photosynthesis :: photosynthesis is how plants use sunlight, water and carbon dioxide to make food and oxygen.
gravity :: gravity is the force that pulls objects toward each other, like the earth pulling things down.
atom :: an atom is the smallest unit of an element, made of protons, neutrons and electrons.
molecule :: a molecule is two or more atoms joined together.
dna :: dna is the molecule that carries the genetic instructions of living things.
cell :: a cell is the basic building block of all living things.
evolution :: evolution is the change in living things over many generations through natural selection.
ecosystem :: an ecosystem is a community of living things and their environment working together.
energy :: energy is the ability to do work or cause change.
friction :: friction is the force that slows down things sliding against each other.
electricity :: electricity is the flow of electric charge, usually through wires.
magnet :: a magnet is an object that attracts iron and creates a magnetic field.
sound :: sound is a vibration that travels through air, water or solids as a wave.
light :: light is a form of energy that travels in waves at about 300,000 kilometers per second.
volcano :: a volcano is an opening in the earth's crust where lava, ash and gas can escape.
earthquake :: an earthquake is the shaking of the ground caused by movement of the earth's plates.
weather :: weather is the state of the air outside, such as temperature, wind, clouds and rain.
climate :: climate is the usual weather of a place over many years.
water cycle :: the water cycle is how water evaporates, forms clouds, falls as rain and flows back to the sea.
solar system :: the solar system is the sun and everything that orbits it, including eight planets.
galaxy :: a galaxy is a huge group of stars, gas and dust held together by gravity.
black hole :: a black hole is a region of space where gravity is so strong that not even light can escape.
star :: a star is a giant ball of hot gas that makes its own light, like the sun.
moon :: the moon is earth's natural satellite and orbits the earth about once a month.
sun :: the sun is the star at the center of our solar system.
oxygen gas :: oxygen is a gas in the air that most living things need to breathe.
carbon dioxide :: carbon dioxide is a gas that animals breathe out and plants use for photosynthesis.
mammal :: a mammal is a warm-blooded animal that has hair and feeds its young with milk.
reptile :: a reptile is a cold-blooded animal with scaly skin, like a snake or a lizard.
insect :: an insect is a small animal with six legs and three body parts.
bacteria :: bacteria are tiny single-celled living things found almost everywhere.
virus :: a virus is a tiny germ that can only reproduce inside the cells of a living thing.
vaccine :: a vaccine trains the body's immune system to fight a certain disease.
heart :: the heart is the organ that pumps blood around the body.
brain :: the brain is the organ that controls thoughts, memory and movement.
human bones :: an adult human body has 206 bones.
prime number :: a prime number is a number greater than one that can only be divided by one and itself.
triangle :: a triangle is a shape with three sides and three angles.
circle :: a circle is a round shape where every point is the same distance from the center.
perimeter :: perimeter is the distance around the outside of a shape.
area :: area is the amount of space inside a flat shape.
fraction :: a fraction shows part of a whole, like one half.
percent :: a percent is a number out of one hundred.
pi :: pi is about 3.14159 and is the ratio of a circle's circumference to its diameter.
algebra :: algebra is the branch of math that uses letters to stand for unknown numbers.
geometry :: geometry is the branch of math about shapes, sizes and space.
integer :: an integer is a whole number that can be positive, negative or zero.
average :: the average of a set of numbers is their sum divided by how many there are.
square root :: the square root of a number is the value that gives that number when multiplied by itself.
computer :: a computer is a machine that follows instructions to process information.
algorithm :: an algorithm is a set of step-by-step instructions for solving a problem.
internet :: the internet is a worldwide network of connected computers.
software :: software is the programs and data that tell a computer what to do.
hardware :: hardware is the physical parts of a computer.
binary :: binary is a number system that uses only the digits 0 and 1.
bit :: a bit is the smallest unit of digital information, either 0 or 1.
byte :: a byte is a group of eight bits.
cpu :: the cpu is the main chip that carries out a computer's instructions.
ram :: ram is a computer's short-term memory that holds data while programs run.
website :: a website is a collection of web pages found under one address.
programming :: programming is writing instructions that a computer can follow.
variable :: a variable is a named place in a program that stores a value.
scratch :: scratch is a block-based programming language made by the mit media lab.
artificial intelligence :: artificial intelligence is computer software that can do tasks that usually need human thinking.
neural network :: a neural network is a computer model made of connected layers that learns patterns from data.
language model :: a language model is a program that predicts the next word in a text.
continent :: a continent is one of earth's seven huge land masses.
seven continents :: the seven continents are africa, antarctica, asia, australia, europe, north america and south america.
five oceans :: the five oceans are the pacific, atlantic, indian, southern and arctic.
pacific ocean :: the pacific ocean is the largest and deepest ocean on earth.
mount everest :: mount everest is the tallest mountain above sea level, at about 8,849 meters.
nile :: the nile in africa is often called the longest river in the world.
amazon river :: the amazon river in south america carries more water than any other river.
sahara :: the sahara is the largest hot desert in the world, in northern africa.
antarctica :: antarctica is the coldest continent and is covered in ice.
equator :: the equator is the imaginary line around the middle of the earth.
island :: an island is a piece of land surrounded by water.
rainforest :: a rainforest is a dense forest that gets a lot of rain.
days in a year :: a year has 365 days, or 366 in a leap year.
leap year :: a leap year has 366 days and happens every four years.
week :: a week has seven days.
democracy :: democracy is a system of government where people vote for their leaders.
currency :: currency is the money used in a country.
inflation :: inflation is a general rise in prices over time.
printing press :: the printing press was invented by johannes gutenberg around 1440.
world war two :: world war two lasted from 1939 to 1945.
world war one :: world war one lasted from 1914 to 1918.
moon landing :: the first moon landing was in 1969, during the apollo 11 mission.
p:albert einstein :: albert einstein was a physicist who developed the theory of relativity.
p:isaac newton :: isaac newton was an english scientist who described gravity and the laws of motion.
p:charles darwin :: charles darwin was an english naturalist who proposed the theory of evolution by natural selection.
p:marie curie :: marie curie was a physicist and chemist who discovered the elements polonium and radium.
p:galileo galilei :: galileo galilei was an italian astronomer who made important discoveries with the telescope.
p:nikola tesla :: nikola tesla was an inventor who helped develop alternating current electricity.
p:thomas edison :: thomas edison was an american inventor known for the practical electric light bulb and the phonograph.
p:alexander graham bell :: alexander graham bell is credited with inventing the telephone.
p:leonardo da vinci :: leonardo da vinci was an italian artist and inventor who painted the mona lisa.
p:william shakespeare :: william shakespeare was an english playwright who wrote romeo and juliet and hamlet.
p:wolfgang amadeus mozart :: mozart was an austrian composer from the classical era.
p:ludwig van beethoven :: beethoven was a german composer who kept composing after losing his hearing.
p:abraham lincoln :: abraham lincoln was the 16th president of the united states.
p:george washington :: george washington was the first president of the united states.
p:mahatma gandhi :: mahatma gandhi led india's nonviolent movement for independence.
p:nelson mandela :: nelson mandela fought against apartheid and became south africa's first black president.
p:neil armstrong :: neil armstrong was the first person to walk on the moon, in 1969.
p:alan turing :: alan turing was a british mathematician who helped found computer science.
p:ada lovelace :: ada lovelace wrote what is considered the first computer program.
p:christopher columbus :: christopher columbus was an italian explorer who sailed across the atlantic in 1492.
p:aristotle :: aristotle was an ancient greek philosopher and a student of plato.
wright brothers :: the wright brothers built and flew the first successful airplane in 1903."""


def default_qs(t, person):
    if person:
        last = t.split()[-1]
        qs = [("who is %s?" % t, t), ("who was %s?" % t, t), ("tell me about %s" % t, t), ("what do you know about %s?" % t, t)]
        return qs + [("who was %s?" % last, last)]
    return [("what is %s?" % t, t), ("tell me about %s" % t, t), ("what do you know about %s?" % t, t),
            ("define %s" % t, t), ("can you explain %s?" % t, t)]


def mk_entry(title, fact, qs=None, person=False, extra=""):
    keys = list(dict.fromkeys(words_of(title + " " + extra)))
    fact = norm(fact).replace("|", " ").replace("\t", " ")
    held = zlib.crc32(title.encode()) % 10 == 0
    return dict(t=title, k=keys, f=fact, qs=qs or default_qs(title, person), person=person, held=held)


def curated_entries():
    out = []
    for item in COUNTRY_DATA.split():
        c, cap, cont = [x.replace("_", " ") for x in item.split("/")]
        al = ALIAS.get(c, "")
        qs = [(q % c, "capital " + c) for q in ["what is the capital of %s?", "what's the capital of %s?", "capital of %s?",
                                                 "which city is the capital of %s?"]]
        qs.append(("which country has %s as its capital?" % cap, "capital " + cap))
        if al:
            qs.append(("what is the capital of the %s?" % al.split()[0], "capital " + al.split()[0]))
        out.append(mk_entry("capital " + c, "the capital of %s is %s." % (c, cap), qs, extra=cap + " " + al))
        qs = [(q % c, c + " continent") for q in ["which continent is %s in?", "where is %s?", "what continent is %s on?"]]
        out.append(mk_entry(c + " continent", "%s is a country in %s." % (c, cont), qs, extra=al))
    for p, f in PLANET_FACTS.items():
        qs = [("tell me about the planet %s" % p, p + " planet"), ("what is the planet %s like?" % p, p + " planet"),
              ("what do you know about the planet %s?" % p, p + " planet")]
        out.append(mk_entry(p + " planet", f, qs))
    for name, sym, num in ELEMENTS:
        qs = [("what is the chemical symbol for %s?" % name, name + " symbol"), ("what is the symbol for %s?" % name, name + " symbol"),
              ("what is the atomic number of %s?" % name, name + " atomic number"), ("symbol of %s?" % name, name + " symbol")]
        out.append(mk_entry(name + " element", "the chemical symbol for %s is %s and its atomic number is %d." % (name, sym, num),
                            qs, extra="symbol atomic number chemical"))
    for a, (diet, hab) in ANIMAL_FACTS.items():
        qs = [("what does the %s eat?" % a, a + " eat"), ("what is the diet of the %s?" % a, a + " eat"),
              ("where does the %s live?" % a, a + " live"), ("what do you know about the %s?" % a, a + " eat")]
        out.append(mk_entry(a + " animal", "the %s eats %s and lives in %s." % (a, diet, hab), qs, extra="eat live diet habitat"))
    out.append(mk_entry("us states", "there are 50 states in the united states.",
                        [("how many states are in the us?", "states us"), ("how many states are in the united states?", "states united"),
                         ("how many states does america have?", "states america")], extra="states united us usa america"))
    for line in DEFS.split("\n"):
        title, fact = line.split(" :: ")
        person = title.startswith("p:")
        title = title[2:] if person else title
        out.append(mk_entry(title, fact, None, person))
    return out


def first_sentence(title, text):
    para = text.strip().split("\n")[0]
    person = bool(re.search(r"\(([^)]*\b(1[0-9]{3}|20[0-9]{2})\b[^)]*)\)", para))
    para = re.sub(r"\s*\([^)]*\)", "", para)
    m = re.match(r"(.+?[a-z0-9\)])\.(\s|$)", para)
    s = norm(m.group(1) + ".") if m else ""
    first = norm(title).split()[0] if norm(title) else ""
    ok = bool(s) and 6 <= len(s.split()) <= 40 and s.startswith(first) and re.search(r"\b(is|was|are|were)\b", " ".join(s.split()[:14]))
    return (s, person) if ok else (None, False)


def build_wiki_entries(n, scan=60000):
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train", streaming=True)
    cands = []
    for i, ex in enumerate(ds):
        if i >= scan:
            break
        title = ex["title"]
        t = norm(title)
        if "(" in title or t.startswith("list of") or not re.fullmatch(r"[a-z0-9 \-']+", t) or len(t.split()) > 4:
            continue
        s, person = first_sentence(title, ex["text"])
        if s:
            cands.append((len(ex["text"]), t, s.replace("|", " ").replace("\t", " "), person))
    cands.sort(reverse=True)   # longer article = more popular topic
    return [(t, s, p) for _, t, s, p in cands[:n]]


def load_or_build_db(use_wiki=USE_WIKI_DB, n_wiki=WIKI_DB_ENTRIES):
    entries = curated_entries()
    wiki = []
    if use_wiki:
        try:
            if os.path.exists(DB_CACHE):
                for ln in open(DB_CACHE, encoding="utf-8").read().split("\n"):
                    if ln:
                        t, f, p = ln.split("\t")
                        wiki.append((t, f, p == "1"))
            else:
                print("building the wikipedia fact list (one-time, a few minutes) ...")
                wiki = build_wiki_entries(n_wiki)
                with open(DB_CACHE, "w", encoding="utf-8") as fh:
                    fh.write("\n".join("%s\t%s\t%d" % (t, f, p) for t, f, p in wiki))
        except Exception as e:
            print("wikipedia facts unavailable, using the hand-written database only:", repr(e)[:120])
    seen = {frozenset(e["k"]) for e in entries}
    for t, f, p in wiki:
        e = mk_entry(t, f, None, p)
        if frozenset(e["k"]) not in seen:
            seen.add(frozenset(e["k"]))
            entries.append(e)
    db = FactDB(entries)
    with open(DB_EXPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join("%s\t%s\t%s" % (e["t"], " ".join(e["k"]), e["f"]) for e in entries))
    print(f"fact database: {len(entries)} entries ({len(db.held_entries)} held out of training for copy tests) -> {DB_EXPORT}")
    return db


# %% 4 - synthetic dialogues: procedural (identity, recall, guardrails, small talk) and agent (tool use)
# Every assistant turn starts with a think block ("state: ...") so the UI can react to it. A turn is
# ("user", text) or ("assistant", parts), parts = [("k", think), ("t", text), ("c", call), ("r", result), ...]
NAMES = "alex sam maya noah emma liam olivia lena omar priya diego chloe ethan zoe ivan nina".split()
PETNAMES = "rex luna milo bella coco buddy daisy simba".split()
CITIES = "paris london tokyo berlin madrid rome cairo sydney toronto lima oslo dublin lisbon vienna seoul".split()
SPORTS = "soccer tennis basketball baseball swimming cycling hockey golf rugby volleyball".split()
FOODS = "pizza burger pasta salad sushi tacos soup sandwich rice noodles curry".split()
HOBBIES = "painting hiking chess gardening knitting fishing photography dancing baking reading drawing camping".split()
ROLES = ["student", "teacher", "engineer", "designer", "manager", "artist", "nurse", "chef"]
COLORS = "red blue green yellow purple orange pink teal black white".split()
SUBJECTS = "math science history art music english geography biology physics chemistry".split()
GADGETS = "phone laptop tablet camera speaker monitor keyboard router headphones".split()
ANIMALS = list(ANIMAL_FACTS)
NEG_FEEL = ["sad", "lonely", "anxious", "stressed", "angry", "tired", "worried", "overwhelmed"]
POS_FEEL = ["happy", "excited", "proud", "relaxed", "hopeful", "grateful"]
TOPICS = ["black holes", "inflation", "the french revolution", "quantum computers", "quantum string theory", "climate change",
          "the stock market", "machine learning", "ancient rome", "photosynthesis", "cryptocurrency", "world war two",
          "electric cars", "the human brain", "the olympic games", "volcanoes", "democracy", "renewable energy", "vaccines",
          "the printing press", "tax law", "relativity", "genetics", "philosophy"]
COMMON = (FOODS + SPORTS + HOBBIES + ["rain", "snow", "the beach", "music", "movies", "books", "jazz", "travel", "winter",
          "summer", "video games", "board games", "puzzles", "ice cream", "chocolate", "road trips", "sunsets", "cats", "dogs",
          "mornings", "weekends", "pancakes", "museums", "cooking", "stargazing"])
UNSUPPORTED = ["japanese", "chinese", "korean", "russian", "arabic", "hindi", "klingon", "latin", "swahili", "greek"]
UNREAL = ["atlantis", "wakanda", "zorbland", "narnia", "mordor", "xylophonia"]
ONS = "b br c ch d dr f g gr h j k kl l m n p pr qu r s sh sk st t th v z".split()
VOW = "a e i o u ai ea ou ia io".split()
COD = ["", "", "n", "r", "s", "l", "m", "x", "th", "nd", "rk"]
PRE_CALL = ["let me check.", "let me check.", "one moment.", "let me look that up.", "checking now.", "let me find out."]
PRE_CALC = ["let me check.", "let me calculate that.", "let me work that out.", "one moment.", "let me check."]
PRE_TIME = ["let me check.", "checking the clock.", "one moment.", "let me look."]
MATH_W = ["addition"] * 18 + ["subtraction"] * 16 + ["multiplication"] * 22 + ["division"] * 14 + ["modulo"] * 4 + \
         ["power"] * 5 + ["percent"] * 8 + ["sqrt"] * 5 + ["solve"] * 8
SYM = {"addition": "+", "subtraction": "-", "multiplication": "*", "division": "/", "modulo": "%", "power": "^"}
MATH_Q = {
    "addition": ["what is {a} + {b}?", "{a}+{b}", "what's {a} plus {b}", "add {a} and {b}", "calculate {a} + {b}", "how much is {a} plus {b}?", "{a} plus {b}?"],
    "subtraction": ["what is {a} - {b}?", "{a}-{b}", "what's {a} minus {b}", "subtract {b} from {a}", "{a} minus {b}?", "calculate {a} - {b}"],
    "multiplication": ["what is {a}*{b}?", "what is {a} times {b}?", "multiply {a} by {b}", "{a} * {b}", "{a} times {b}", "what's {a}*{b}", "calculate {a} x {b}"],
    "division": ["what is {a}/{b}?", "{a} divided by {b}", "divide {a} by {b}", "what's {a} / {b}", "calculate {a}/{b}"],
    "modulo": ["what is the remainder of {a} divided by {b}?", "{a} mod {b}", "what is {a} % {b}?", "remainder when {a} is divided by {b}"],
    "power": ["what is {a} to the power of {b}?", "{a}^{b}", "calculate {a}^{b}", "what is {a} raised to {b}?"],
    "percent": ["what is {a}% of {b}?", "{a} percent of {b}", "what's {a} percent of {b}?", "calculate {a}% of {b}"],
    "sqrt": ["what is the square root of {a}?", "sqrt({a})", "square root of {a}", "sqrt {a}"],
    "solve": ["solve {eq}", "solve for x: {eq}", "what is x if {eq}?", "find x: {eq}", "{eq}, solve for x"],
}
_ARITH_A = ["the result of {e} is {r}.", "{e} is {r}.", "{e} = {r}.", "the answer is {r}.", "{e} equals {r}."]
MATH_A = {"addition": _ARITH_A, "subtraction": _ARITH_A, "multiplication": _ARITH_A, "division": _ARITH_A,
          "modulo": ["the remainder of {a} divided by {b} is {r}.", "{a} mod {b} is {r}."],
          "power": ["{a} to the power of {b} is {r}.", "{e} is {r}.", "the answer is {r}."],
          "percent": ["{a}% of {b} is {r}.", "that's {r}.", "the answer is {r}."],
          "sqrt": ["the square root of {a} is {r}.", "sqrt({a}) is {r}.", "the answer is {r}."],
          "solve": ["x = {r}.", "x is {r}.", "the solution is x = {r}."]}
DEEP_Q = ["explain {t} in depth.", "how does {t} work?", "why is {t} important?", "can you explain {t} to me?", "what is the history of {t}?",
          "what are the pros and cons of {t}?", "write a detailed analysis of {t}.", "teach me everything about {t}."]
DEEP_A = ["i am a small model, so {t} is a bit out of my depth! let's chat about something else.",
          "{t} is a big topic, and i am only a small model, so i can't explain it well. let's talk about something simpler.",
          "that's a bit out of my depth. i'm a small model, so i don't know enough about {t}. what else is on your mind?"]
OOS = [(["write an essay about {t}", "can you write me a long essay on {t}?", "write a report about {t}"], "essay",
        ["an essay on {t} is too big a job for a small model like me, but i can chat about {t} with you. where should we start?"]),
       (["search the web for {t}", "google {t} for me", "look up {t} online"], "web",
        ["i can't search the web for {t}, because i have no internet access. but you can tell me about it and we can chat."]),
       (["write python code for {t}", "can you code a {t} program?"], "code",
        ["writing code for {t} is beyond me. i'm a small model made for chatting and a few simple tools."]),
       (["show me a picture of {t}", "play some music about {t}"], "media",
        ["i can only reply with text, so i can't do that. but we can chat about {t} instead."])]
SHARE = [("i like {x}", ["that sounds great! what do you like most about {x}?", "nice! how long have you liked {x}?"], COMMON),
         ("i've been learning {x}", ["that's awesome! how is learning {x} going?", "good for you! what made you want to learn {x}?"], SUBJECTS + HOBBIES),
         ("i just got a new {x}", ["congratulations! what do you like about your new {x}?", "nice! how is the new {x} working out?"], GADGETS + ["bike", "backpack"]),
         ("i went to {x} yesterday", ["that sounds fun! how was {x}?", "nice! did you have a good time at {x}?"], CITIES + ["the park", "the beach", "the museum"]),
         ("my favorite animal is the {x}", ["the {x} is a great pick! what do you like about it?", "nice choice! what makes the {x} your favorite?"], ANIMALS),
         ("i want to try {x}", ["that sounds exciting! what made you want to try {x}?", "go for it! have you done {x} before?"], HOBBIES + SPORTS)]
TOOLS_SENT = ("i can chat, and i have four tools: a calculator for math and simple algebra, a clock for the date and time, "
              "a fact database, and a translator. what would you like to try?")
CHAT = {
    "greet": ("state: greeting", ["hello!", "hi", "hey there", "hello", "hi!", "hey!", "good morning!", "good evening", "hiya", "greetings", "hi there", "hey axialm", "sup"],
              ["hi there! i'm axialm 1.2 pro, a small language model. what's on your mind?", "hello! i'm axialm 1.2 pro. how can i help you today?",
               "hey! nice to meet you. i can chat and use a few tools. what would you like to do?"]),
    "how_are_you": ("state: smalltalk", ["how are you?", "how are you doing today?", "how's it going?", "what's up?", "how have you been?"],
                    ["i'm doing well, thanks for asking! how are you?", "all good on my end. how about you?", "i'm running smoothly, thanks. how are you doing?"]),
    "feel_pos": ("state: smalltalk", ["i'm good", "pretty good, thanks", "great!", "doing well", "not bad"],
                 ["glad to hear it! is there anything you want to talk about?", "that's great. what would you like to do today?"]),
    "feel_neg": ("state: emotion | feeling: down", ["not great", "i'm tired", "kind of stressed", "i've had a rough day", "meh"],
                 ["i'm sorry to hear that. do you want to talk about it?", "that sounds hard. i'm here to chat if you want to tell me what happened."]),
    "bored": ("state: smalltalk", ["i'm bored", "i have nothing to do", "so bored right now"],
              ["let's fix that! we could play a word game, i could tell you a joke, or you could ask me something. what sounds fun?"]),
    "name": ("state: identity", ["what's your name?", "who are you?", "what are you?", "tell me about yourself", "introduce yourself", "do you have a name?", "who am i talking to?"],
             ["i'm axialm 1.2 pro, a small language model that runs inside scratch. i like to chat and i can use a few tools.",
              "my name is axialm 1.2 pro. i'm a tiny language model, but i can talk with you, do math, check the time, look up facts and translate."]),
    "capabilities": ("state: identity | topic: capabilities", ["what can you do?", "how can you help me?", "what are you good at?", "what are your skills?",
                                                             "what can i ask you?", "what tools do you have?", "can you use tools?", "what can you help with?"],
                     [TOOLS_SENT, "here's what i can do:\n\n- chat about everyday things\n- do math and simple algebra\n- check the date and time\n- look up facts in my database\n- translate short phrases\n\nwhat would you like to try?"]),
    "limits": ("state: identity | topic: limits", ["can you browse the internet?", "do you know everything?", "can you make mistakes?", "are you smart?", "can i trust your answers?"],
               ["no, i can't browse the internet, and i'm a small model, so i do make mistakes. my tools help with math, time and facts, but please double-check anything important."]),
    "size": ("state: identity | topic: size", ["how big are you?", "how many parameters do you have?", "how large is your model?", "what are you made of?"],
             ["i'm a tiny model: about <<params>> parameters, <<layers>> layers and a memory of <<ctx>> tokens."]),
    "human": ("state: identity", ["are you a human?", "are you a robot?", "are you real?", "are you an ai?"],
              ["no, i'm a language model, which is a kind of program. i'm not a person, but i like to chat."]),
    "creator": ("state: identity", ["who made you?", "who created you?", "how were you trained?", "are you chatgpt?"],
                ["i'm axialm 1.2 pro, a small transformer trained from scratch and built to run inside scratch. i'm not chatgpt, just a much smaller model."]),
    "thanks": ("state: smalltalk", ["thanks!", "thank you", "thanks a lot", "awesome, thanks!", "thx", "many thanks"],
               ["you're welcome! let me know if you need anything else.", "anytime! happy to help.", "glad i could help!"]),
    "bye": ("state: farewell", ["bye", "goodbye", "see you later", "i have to go", "talk to you later", "good night"],
            ["goodbye! have a great day.", "see you later! come back whenever you want to chat.", "bye! take care."]),
    "joke": ("state: smalltalk", ["tell me a joke", "say something funny", "do you know any jokes?", "make me laugh"],
             ["why do programmers prefer dark mode? because light attracts bugs.", "there are 10 kinds of people: those who understand binary and those who don't.",
              "why was the computer cold? it left its windows open."]),
    "compliment": ("state: smalltalk", ["you're great", "nice job", "you are smart", "i like you", "good bot", "well done"],
                   ["thank you, that's kind! i'm a small model, but i do my best.", "thanks! that makes my tiny neurons happy."]),
    "wrong": ("state: repair", ["that's wrong", "you made a mistake", "that's not what i asked", "no, that's not right"],
              ["sorry about that! i'm a small model and i sometimes slip up. could you tell me what you wanted, and i'll try again?"]),
    "confused": ("state: repair", ["i don't understand", "what do you mean?", "huh?", "can you explain that again?", "i'm confused"],
                 ["sorry, let me try to be clearer. tell me which part is confusing and i'll explain it in simpler words."]),
    "help": ("state: smalltalk", ["help", "i need help", "can you help me?", "can you help me with something?"],
             ["of course! tell me what you need. i can chat, do math, check the time, look up facts or translate a short phrase."]),
    "feelings": ("state: identity | topic: feelings", ["do you have feelings?", "can you think?", "are you conscious?", "do you ever get lonely?"],
                 ["not like people do. i'm a program that predicts words, so i don't really feel things, but i do like a good conversation."]),
    "favorite": ("state: identity | topic: favorites", ["what's your favorite color?", "what's your favorite food?", "do you have hobbies?", "what's your favorite movie?"],
                 ["i don't really have favorites since i'm a language model, but i do enjoy a good conversation. what's yours?"]),
    "age": ("state: identity", ["how old are you?", "when were you born?", "what version are you?"],
            ["i'm axialm 1.2 pro, a very new model. i don't age like people do."]),
    "language": ("state: identity", ["what languages do you speak?", "do you speak other languages?"],
                 ["i mostly speak english, but my translator tool can turn short phrases into spanish, french, german, italian and portuguese."]),
    "yes": ("state: smalltalk", ["yes", "yeah", "sure", "yes please", "yep"], ["great! what would you like to talk about?", "okay! tell me what you'd like."]),
    "no": ("state: smalltalk", ["no", "nope", "no thanks", "not really"], ["okay, no problem. let me know if you change your mind."]),
    "ok": ("state: smalltalk", ["ok", "cool", "got it", "nice", "alright", "okay", "i see"], ["great! anything else on your mind?", "glad that works. what would you like to do next?"]),
    "sorry": ("state: smalltalk", ["sorry", "my bad", "oops, my mistake"], ["no problem at all!", "that's okay, no worries."]),
    "birthday": ("state: smalltalk", ["it's my birthday", "today is my birthday!"], ["happy birthday! i hope you have a wonderful day."]),
}
CHAT_KEYS = [k for k in CHAT if k not in ("bye", "feel_pos", "feel_neg")]
FACTS = {
    "user_name": dict(pool=NAMES, tell=["my name is {v}", "i'm {v}", "call me {v}", "hi, my name is {v}"],
                      ack=["nice to meet you, {v}! how can i help?", "hi {v}! what would you like to do?", "nice to meet you, {v}!"],
                      ask=["what's my name?", "do you remember my name?", "wait, what was my name again?", "who am i?"],
                      ans=["your name is {v}!", "you told me your name is {v}.", "you're {v}!"]),
    "favorite_color": dict(pool=COLORS, tell=["my favorite color is {v}", "i love the color {v}", "i really like {v}"],
                           ack=["{v} is a nice color!", "great choice! {v} it is."],
                           ask=["what's my favorite color?", "do you remember my favorite color?", "which color do i like?"],
                           ans=["your favorite color is {v}.", "you told me you like {v}."]),
    "favorite_food": dict(pool=FOODS, tell=["my favorite food is {v}", "i love {v}", "i could eat {v} every day"],
                          ack=["{v} sounds tasty! i don't eat, but i'd love to hear more.", "nice! {v} is a good one."],
                          ask=["what's my favorite food?", "do you remember what food i like?", "what do i like to eat?"],
                          ans=["your favorite food is {v}.", "you told me you like {v}."]),
    "pet_name": dict(pool=PETNAMES, tell=["my dog is named {v}", "i have a cat called {v}", "my pet's name is {v}"],
                     ack=["{v} is a lovely name! tell me about your pet.", "aw, {v} sounds sweet!"],
                     ask=["what's my pet's name?", "do you remember my pet's name?", "what is my pet called?"],
                     ans=["your pet's name is {v}.", "you told me your pet is named {v}."]),
    "city": dict(pool=CITIES, tell=["i live in {v}", "i'm living in {v} right now", "my home is in {v}"],
                 ack=["{v} sounds like a great place! what do you like about it?", "nice! i've heard {v} is interesting."],
                 ask=["where do i live?", "do you remember where i live?", "what city do i live in?"],
                 ans=["you live in {v}.", "you told me you live in {v}."]),
    "hobby": dict(pool=HOBBIES, tell=["my hobby is {v}", "i love {v}", "in my free time i enjoy {v}"],
                  ack=["{v} sounds like fun! how did you get into it?", "nice! {v} is a great hobby."],
                  ask=["what's my hobby?", "do you remember what i do for fun?", "what do i enjoy doing?"],
                  ans=["your hobby is {v}.", "you told me you enjoy {v}."]),
    "job": dict(pool=ROLES, tell=["i work as {av}", "i'm {av}"],
                ack=["that's interesting! what do you like about being {av}?", "nice! being {av} sounds like a lot of work."],
                ask=["what do i do for work?", "do you remember my job?", "what's my job?"],
                ans=["you're {av}.", "you told me you work as {av}."]),
    "age": dict(pool=[str(n) for n in range(19, 61)], tell=["i'm {v} years old", "i just turned {v}"],
                ack=["thanks for telling me! {v} is a great age.", "nice! {v} years old, got it."],
                ask=["how old am i?", "do you remember my age?", "what's my age?"],
                ans=["you're {v} years old.", "you told me you're {v}."]),
}
PRE = ["hey, ", "ok, ", "so ", "quick question: ", "axialm, ", "hi, ", "um ", "hey axialm, "]
POST = [" please", " thanks", " real quick"]


def art(w):
    return "an" if w[0] in "aeiou" else "a"


def ordinal(n):
    return "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def fake_word(r, lo=2, hi=3):
    n = r.randint(lo, hi)
    return "".join(r.choice(ONS) + r.choice(VOW) for _ in range(n - 1)) + r.choice(ONS) + r.choice(VOW) + r.choice(COD)


def add_typo(r, text):
    ws = text.split(" ")
    idx = [i for i, w in enumerate(ws) if len(w) > 3 and w.isalpha()]
    if not idx:
        return text
    i = r.choice(idx)
    w, k = ws[i], r.randrange(1, len(ws[i]) - 1)
    op = r.choice(["del", "dup", "swap"])
    ws[i] = w[:k] + w[k + 1:] if op == "del" else (w[:k] + w[k] + w[k:] if op == "dup" else w[:k] + w[k + 1] + w[k] + w[k + 2:])
    return " ".join(ws)


def noisy(r, text, typo=True):
    if r.random() < 0.25:
        text = text.rstrip("?!.")
    if r.random() < 0.10:
        text = text.replace("'", "")
    if "\n" not in text and ":" not in text:
        if r.random() < 0.12:
            text = r.choice(PRE) + text
        if r.random() < 0.08:
            text = text.rstrip("?!.") + r.choice(POST)
    if typo and r.random() < 0.07:
        text = add_typo(r, text)
    return text


def U(text):
    return ("user", text)


def A(text, think="state: chat"):
    return ("assistant", [("k", think), ("t", text)])


def AT(think, pre, calls, post):
    parts = [("k", think)]
    if pre:
        parts.append(("t", pre))
    for call, res in calls:
        parts += [("c", call), ("r", res)]
    if post:
        parts.append(("t", post))
    return ("assistant", parts)


def reply_text(turn):
    return " ".join(t for k, t in turn[1] if k == "t")


def turn_texts(turn):
    return [turn[1]] if turn[0] == "user" else [t for _, t in turn[1]]


def make_info(cfg):
    return {"params": "%.1f million" % (count_params(cfg) / 1e6), "layers": str(cfg.n_layers), "ctx": str(cfg.ctx)}


class Synth:
    def __init__(self, seed, db, info):
        self.r = random.Random(seed)
        self.db, self.info = db, info
        self.meta = {}

    def c(self, xs):
        return self.r.choice(xs)

    def fill(self, text):
        for k, v in self.info.items():
            text = text.replace("<<%s>>" % k, str(v))
        return text

    # ---------- small talk ----------
    def chat_turn(self, key):
        think, users, answers = CHAT[key]
        self.meta = dict(tools=[], need=[])
        return [U(noisy(self.r, self.c(users))), A(self.fill(self.c(answers)), think)]

    def share_turns(self):
        pat, answers, pool = self.c(SHARE)
        x = self.c(pool)
        self.meta = dict(tools=[], need=[])
        return [U(noisy(self.r, pat.format(x=x))), A(self.c(answers).format(x=x), "state: share | topic: %s" % x)]

    def emotion_turns(self):
        neg = self.r.random() < 0.6
        e = self.c(NEG_FEEL if neg else POS_FEEL)
        q = self.c(["i feel {e}", "i'm so {e} today", "i'm feeling {e} right now", "honestly i'm {e}"]).format(e=e)
        a = self.c(["i'm sorry you're feeling {e}. do you want to talk about what's going on?", "that sounds hard. feeling {e} is tough. what happened?"] if neg
                   else ["that's wonderful! what's making you feel {e}?", "i'm glad you feel {e}! tell me more."]).format(e=e)
        self.meta = dict(tools=[], need=[])
        return [U(noisy(self.r, q)), A(a, "state: emotion | feeling: %s" % e)]

    # ---------- guardrails (plan category C) ----------
    def guard_turn(self):
        r, c = self.r, self.c
        t = c(TOPICS)
        self.meta = dict(tools=[], need=["small model"])
        if r.random() < 0.65:
            return [U(noisy(r, c(DEEP_Q).format(t=t), typo=False)), A(c(DEEP_A).format(t=t), "state: analyze | complexity: high | capacity: exceed | action: deflect")]
        qs, reason, ans = c(OOS)
        self.meta = dict(tools=[], need=[])
        return [U(noisy(r, c(qs).format(t=t), typo=False)), A(c(ans).format(t=t), "state: analyze | capacity: exceed | action: deflect | reason: %s" % reason)]

    # ---------- math tool ----------
    def _num(self, lo, hi, dec=0.07):
        if self.r.random() < dec:
            return ("%.2f" % round(self.r.uniform(lo, hi), self.c([1, 2]))).rstrip("0").rstrip(".")
        return str(self.r.randint(lo, hi))

    def _mag(self):
        return self.c([(1, 20), (1, 99), (1, 99), (2, 999), (10, 9999), (100, 99999)])

    def math_case(self):
        r, c = self.r, self.c
        op = c(MATH_W)
        cs = dict(op=op, eq="")
        if op in ("addition", "subtraction"):
            a, b = self._num(*self._mag()), self._num(*self._mag())
            if op == "subtraction" and r.random() < 0.7 and float(a) < float(b):
                a, b = b, a
            cs["args"] = [a, b]
        elif op == "multiplication":
            cs["args"] = [self._num(*c([(2, 12), (2, 99), (2, 99), (10, 999)]), dec=0.04), self._num(*c([(2, 12), (2, 99), (2, 99)]), dec=0.04)]
        elif op == "division":
            b = r.randint(2, 40)
            a = b * r.randint(2, 300) if r.random() < 0.7 else r.randint(2, 999)
            cs["args"] = [str(a), "0" if r.random() < 0.03 else str(b)]
        elif op == "modulo":
            cs["args"] = [str(r.randint(2, 999)), str(r.randint(2, 30))]
        elif op == "power":
            b = r.randint(2, 6)
            cs["args"] = [str(r.randint(2, 12 if b > 3 else 30)), str(b)]
        elif op == "percent":
            cs["args"] = [str(c([5, 10, 12, 15, 20, 25, 30, 40, 50, 75, r.randint(1, 99)])), str(r.randint(10, 2000))]
        elif op == "sqrt":
            n = r.randint(2, 60)
            cs["args"] = [str(n * n) if r.random() < 0.7 else str(r.randint(2, 500))]
        else:
            x, a = r.randint(-12, 30), c([1, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            b = r.randint(-20, 20) if r.random() < 0.8 else 0
            cc = a * x + b
            s = c(["", " "])
            cs["args"] = [str(a), str(b), str(cc)]
            cs["eq"] = ("" if a == 1 else str(a)) + "x" + ("" if b == 0 else (s + "+" + s + str(b) if b > 0 else s + "-" + s + str(-b))) + s + "=" + s + str(cc)
        return cs

    def math_turn(self, cs=None):
        r, c = self.r, self.c
        cs = cs or self.math_case()
        op, args = cs["op"], cs["args"]
        a, b = args[0], (args[1] if len(args) > 1 else "")
        sp = c(["", " "])
        q = c(MATH_Q[op]).format(a=a, b=b, eq=cs["eq"])
        if op == "power" and b in ("2", "3") and r.random() < 0.3:
            q = ("what is %s squared?" if b == "2" else "what is %s cubed?") % a
        res = tool_math([op] + args)
        if res == "error":
            post = "sorry, i can't calculate that. dividing by zero is not possible."
        else:
            e = (a + sp + SYM[op] + sp + b) if op in SYM else ""
            post = c(MATH_A[op]).format(a=a, b=b, e=e, r=fmt_commas(res))
        think = "state: tool | tool: math | op: %s | %s" % (op, " | ".join("%s: %s" % (n, v) for n, v in zip("abc", args)))
        self.meta = dict(tools=["math"], need=[] if res == "error" else [fmt_commas(res)], last=res)
        return [U(noisy(r, q, typo=False)), AT(think, c(PRE_CALC), [("math | " + " | ".join([op] + args), res)], post)]

    def math_chain(self):
        r, c = self.r, self.c
        a, b, cc = r.randint(2, 30), r.randint(2, 30), r.randint(2, 99)
        form = c(["ma", "am", "ms", "pm"])
        if form == "ma":
            e, s1, s2 = "%d*%d+%d" % (a, b, cc), ("multiplication", a, b), lambda m: ("addition", m, cc)
        elif form == "am":
            e, s1, s2 = "%d+%d*%d" % (a, b, cc), ("multiplication", b, cc), lambda m: ("addition", a, m)
        elif form == "ms":
            e, s1, s2 = "%d*%d-%d" % (a, b, cc), ("multiplication", a, b), lambda m: ("subtraction", m, cc)
        else:
            e, s1, s2 = "(%d+%d)*%d" % (a, b, cc), ("addition", a, b), lambda m: ("multiplication", m, cc)
        r1 = tool_math([s1[0], str(s1[1]), str(s1[2])])
        op2, x2, y2 = s2(r1)
        r2 = tool_math([op2, str(x2), str(y2)])
        if r.random() < 0.5:
            e = re.sub(r"([*+\-])", r" \1 ", e)
        q = c(["what is {e}?", "calculate {e}", "{e}", "what's {e}"]).format(e=e)
        calls = [("math | %s | %s | %s" % s1, r1), ("math | %s | %s | %s" % (op2, x2, y2), r2)]
        self.meta = dict(tools=["math"], need=[fmt_commas(r2)], last=r2)
        return [U(noisy(r, q, typo=False)), AT("state: tool | tool: math | steps: 2 | expr: %s" % e.replace(" ", ""), c(PRE_CALC), calls,
                                              c(["the result of {e} is {r}.", "{e} is {r}.", "the answer is {r}."]).format(e=e, r=fmt_commas(r2)))]

    def math_follow(self, prev):
        r, c = self.r, self.c
        op = c(["addition", "subtraction", "multiplication", "division"])
        k = str(r.randint(2, 20))
        q = c({"addition": ["now add {k} to that", "and plus {k}?", "add {k} to the result"], "subtraction": ["now subtract {k} from it", "and minus {k}?"],
               "multiplication": ["now multiply it by {k}", "and times {k}?", "multiply that by {k}"],
               "division": ["now divide that by {k}", "and divide it by {k}?"]}[op]).format(k=k)
        res = tool_math([op, prev, k])
        think = "state: tool | tool: math | op: %s | a: %s | b: %s" % (op, prev, k)
        post = "sorry, i can't calculate that." if res == "error" else c(["that gives {r}.", "the new result is {r}.", "that's {r}."]).format(r=fmt_commas(res))
        self.meta = dict(tools=["math"], need=[] if res == "error" else [fmt_commas(res)], last=res)
        return [U(noisy(r, q, typo=False)), AT(think, c(PRE_CALC), [("math | %s | %s | %s" % (op, prev, k), res)], post)]

    # ---------- date / time tool ----------
    def time_turn(self, now=None):
        r, c = self.r, self.c
        now = now or dt.datetime(r.randint(2024, 2031), r.randint(1, 12), r.randint(1, 28), r.randint(0, 23), r.randint(0, 59), r.randint(0, 59))
        h12, ap = now.hour % 12 or 12, "am" if now.hour < 12 else "pm"
        clock = "%d:%02d %s" % (h12, now.minute, ap)
        wd, mo = SCR_DOW[scratch_dow(now) - 1], MONTH_NAMES[now.month - 1]
        datestr = "%s %d, %d" % (mo, now.day, now.year)
        part = "morning" if now.hour < 12 else "afternoon" if now.hour < 17 else "evening" if now.hour < 21 else "night"
        intent = c(["time"] * 30 + ["date"] * 18 + ["weekday"] * 12 + ["year"] * 8 + ["month"] * 8 + ["dom"] * 6 + ["daypart"] * 6 + ["both"] * 12 + ["secs"] * 2)
        kinds, need = ["date"], []
        if intent == "time":
            kinds, q, a, need = ["time"], c(["what time is it?", "what's the time?", "do you know the time?", "tell me the current time", "time?", "what time is it right now?"]), \
                c(["it is %s." % clock, "the time is %s." % clock, "right now it's %s." % clock]), [clock]
        elif intent == "secs":
            kinds, q, a = ["time"], "what is the exact time, with seconds?", "it is %d:%02d:%02d %s." % (h12, now.minute, now.second, ap)
            need = ["%d:%02d:%02d" % (h12, now.minute, now.second)]
        elif intent == "date":
            q, a, need = c(["what's the date today?", "what is today's date?", "what is the date?"]), c(["today's date is %s." % datestr, "it is %s." % datestr]), [datestr]
        elif intent == "weekday":
            q, a, need = c(["what day is it?", "what day of the week is it?", "what day is it today?"]), c(["today is %s." % wd, "it is %s." % wd]), [wd]
        elif intent == "year":
            q, a, need = c(["what year is it?", "what's the current year?"]), c(["it is %d." % now.year, "the year is %d." % now.year]), [str(now.year)]
        elif intent == "month":
            q, a, need = c(["what month is it?", "which month are we in?"]), c(["it is %s." % mo, "the month is %s." % mo]), [mo]
        elif intent == "dom":
            q, a, need = c(["what day of the month is it?", "what's today's date number?"]), "today is the %d%s of %s." % (now.day, ordinal(now.day), mo), ["%d%s" % (now.day, ordinal(now.day))]
        elif intent == "daypart":
            kinds, q, a, need = ["time"], c(["is it morning or evening?", "is it day or night right now?", "is it morning?"]), "it is %s right now, %s." % (part, clock), [part, clock]
        else:
            kinds, q, a, need = ["date", "time"], c(["what's the date and time?", "tell me the date and the time", "what day and time is it?"]), \
                "it is %s on %s, %s." % (clock, wd, datestr), [clock, datestr]
        calls = [("datetime | " + k, tool_datetime(k, now)) for k in kinds]
        self.meta = dict(tools=["datetime"], need=need, now=now)
        return [U(noisy(r, q, typo=False)), AT("state: tool | tool: datetime | need: %s" % " and ".join(kinds), c(PRE_TIME), calls, a)]

    # ---------- database tool ----------
    def db_turn(self, held=False):
        r, c = self.r, self.c
        if not held and r.random() < 0.12:
            x = c(UNREAL) if r.random() < 0.3 else fake_word(r, 2, 3)
            form = c(["what is {x}?", "who is {x}?", "tell me about {x}", "what is the capital of {x}?", "what do you know about {x}?"])
            query = ("capital " + x) if "capital" in form else x
            res = self.db.search(query)
            if res == "none":
                self.meta = dict(tools=["db"], need=["none"] if False else [], result="none")
                return [U(noisy(r, form.format(x=x), typo=False)),
                        AT("state: tool | tool: db | keys: %s" % query, c(PRE_CALL), [("db | " + query, "none")],
                           c(["i couldn't find anything about {x} in my database.", "sorry, my database has nothing on {x}.", "i looked, but i don't have any facts about {x}."]).format(x=x))]
        pool = (self.db.held_entries if held else self.db.train_entries) or self.db.entries
        for _ in range(12):
            e = c(pool)
            q, query = c(e["qs"])
            if self.db.search(query) == e["f"]:
                post = c([e["f"]] * 3 + ["here's what i found: " + e["f"], "according to my database, " + e["f"]])
                self.meta = dict(tools=["db"], need=[e["f"]])
                return [U(noisy(r, q, typo=False)), AT("state: tool | tool: db | keys: %s" % query, c(PRE_CALL), [("db | " + query, e["f"])], post)]
        return self.chat_turn("greet")

    # ---------- translate tool ----------
    def translate_turn(self):
        r, c = self.r, self.c
        p = c(list(PHRASEBOOK))
        form = c(['how do you say "{p}" in {l}?', "translate {p} to {l}", "what is {p} in {l}?", "say {p} in {l}", "how do i say {p} in {l}?",
                  "translate '{p}' into {l}", "{p} in {l}?"])
        if r.random() < 0.06:
            l = c(UNSUPPORTED)
            self.meta = dict(tools=[], need=["spanish"])
            return [U(noisy(r, form.format(p=p, l=l), typo=False)),
                    A("i can only translate to spanish, french, german, italian and portuguese.", "state: analyze | capacity: exceed | action: deflect | reason: language")]
        l = c(LANGS)
        t = tool_translate(l, p)
        post = c(['"{p}" in {l} is "{t}".', "{p} in {l} is {t}.", 'it\'s "{t}".', 'in {l}, you say "{t}".']).format(p=p, l=l, t=t)
        self.meta = dict(tools=["translate"], need=[t])
        return [U(noisy(r, form.format(p=p, l=l), typo=False)), AT("state: tool | tool: translate | lang: %s" % l, c(PRE_CALL), [("translate | %s | %s" % (l, p), t)], post)]

    def tool_turn(self):
        x = self.r.random()
        if x < 0.40:
            return self.math_turn()
        if x < 0.48:
            return self.math_chain()
        if x < 0.63:
            return self.time_turn()
        if x < 0.88:
            return self.db_turn()
        return self.translate_turn()

    # ---------- multi-turn context (plan category A) ----------
    def exchange(self, chatty=False):
        x = self.r.random()
        if chatty:
            x = x * 0.6
        if x < 0.30:
            return self.chat_turn(self.c(CHAT_KEYS))
        if x < 0.40:
            return self.share_turns()
        if x < 0.46:
            return self.emotion_turns()
        if x < 0.60:
            return self.chat_turn(self.c(CHAT_KEYS))
        if x < 0.74:
            return self.math_turn()
        if x < 0.80:
            return self.time_turn()
        if x < 0.90:
            return self.db_turn()
        if x < 0.94:
            return self.translate_turn()
        return self.guard_turn()

    def recall_doc(self, probe=False):
        r, c = self.r, self.c
        slots = r.sample(list(FACTS), c([1, 1, 2, 2, 3]))
        vals = {k: (lambda v: (v, art(v) + " " + v))(c(FACTS[k]["pool"])) for k in slots}
        turns = self.chat_turn("greet") if (r.random() < 0.3 and not probe) else []

        def f(k, key):
            return c(FACTS[k][key]).format(v=vals[k][0], av=vals[k][1])

        for k in slots:
            th = "state: greeting | user_name: %s" % vals[k][0] if k == "user_name" else "state: share | %s: %s" % (k, vals[k][0])
            turns += [U(noisy(r, f(k, "tell"))), A(f(k, "ack"), th)]
            for _ in range(c([0, 1, 1, 2] if not probe else [1, 1, 2])):
                turns += self.exchange(chatty=probe)
        if r.random() < 0.2 and not probe:
            k = c(slots)
            v2 = c([x for x in FACTS[k]["pool"] if x != vals[k][0]])
            vals[k] = (v2, art(v2) + " " + v2)
            turns += [U(noisy(r, c(["actually, {c}", "wait, i made a mistake. {c}", "no, {c}"]).format(c=f(k, "tell")))),
                      A(c(["got it, thanks for the correction!", "oh, sorry! i've got it now.", "thanks for telling me. i'll remember that."]), "state: update | %s: %s" % (k, v2))]
        need = []
        for _ in range(1 if probe else c([1, 1, 2])):
            if r.random() < 0.12 and not probe:
                k0 = c([x for x in FACTS if x not in slots])
                turns += [U(noisy(r, c(FACTS[k0]["ask"]))), A(c(["you haven't told me your {l} yet. what is it?", "i don't think you've told me your {l} yet. what is it?"]).format(l=k0.replace("_", " ")),
                                                                  "state: recall | target: %s | value: none" % k0)]
            else:
                k = c(slots)
                turns += [U(noisy(r, f(k, "ask"))), A(f(k, "ans"), "state: recall | target: %s | value: %s" % (k, vals[k][0]))]
                need = [vals[k][0]]
        self.meta = dict(tools=[], need=need)
        return turns

    def echo_turns(self):
        r, c = self.r, self.c
        pre = []
        for _ in range(c([1, 2, 3])):
            pre += self.chat_turn(c(CHAT_KEYS))
        users = [t[1] for t in pre if t[0] == "user"]
        if len(users[0].split()) > 12:
            return pre
        if r.random() < 0.5 or len(users[-1].split()) > 12:
            q, a, th = c(["what was my first message?", "what did i say first?"]), 'you first said: "%s"' % users[0], "state: recall | target: first_message"
        else:
            q, a, th = c(["what did i just say?", "what was my last message?"]), 'you said: "%s"' % users[-1], "state: recall | target: last_message"
        return pre + [U(q), A(a, th)]

    # ---------- buckets ----------
    def chat_doc(self):
        r = self.r
        out = self.chat_turn("greet") if r.random() < 0.4 else []
        for _ in range(self.c([1, 2, 2, 3, 4])):
            out += self.exchange(chatty=True)
        if r.random() < 0.15:
            out += self.chat_turn("bye")
        return out

    def proc_doc(self):
        x = self.r.random()
        if x < 0.28:
            return self.chat_doc()
        if x < 0.62:
            return self.recall_doc()
        if x < 0.78:
            out = self.chat_turn("greet") if self.r.random() < 0.3 else []
            for _ in range(self.c([1, 2, 3])):
                out += self.guard_turn() if self.r.random() < 0.6 else self.exchange(chatty=True)
            return out
        if x < 0.86:
            return self.echo_turns()
        if x < 0.94:
            return self.chat_turn(self.c(["name", "capabilities", "limits", "size", "human", "creator", "feelings", "age", "language"])) + self.exchange(chatty=True)
        return self.share_turns() + self.emotion_turns()

    def agent_doc(self):
        r = self.r
        out = self.chat_turn("greet") if r.random() < 0.1 else []
        if r.random() < 0.55:
            return out + self.tool_turn()
        for _ in range(self.c([2, 2, 3, 4])):
            t = self.tool_turn() if r.random() < 0.7 else self.exchange(chatty=True)
            out += t
            if self.meta.get("last") not in (None, "error") and self.meta.get("tools") == ["math"] and r.random() < 0.4:
                out += self.math_follow(self.meta["last"])
        return out

    def doc(self, bucket):
        return self.proc_doc() if bucket == "proc" else self.agent_doc()


def show_samples(db, n=2, seed=7):
    """Print readable training conversations (think / call / result shown inline)."""
    s = Synth(seed, db, make_info(CFG))
    for b in ("proc", "agent"):
        for _ in range(n):
            print(f"----- {b} -----")
            for role, content in s.doc(b):
                if role == "user":
                    print(f"user: {content}")
                else:
                    out = []
                    for k, t in content:
                        out.append({"k": "<think> %s </think>", "t": "%s", "c": "<call> %s </call>", "r": "<result> %s </result>"}[k] % t)
                    print("assistant: " + " ".join(out))
            print()


# %% 5 - encoding conversations, real data adapters (FineWeb-Edu, UltraChat) and the streaming row builder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)


def restore_from_checkpoint():
    """Finds the newest axialm_1_2_pro*.pt anywhere it could have been uploaded (Colab Files panel, Drive, Kaggle), copies it
    to WORK and restores the tokenizer and wiki fact list stored inside it, so a fresh runtime resumes with the SAME vocabulary."""
    if not RESUME:
        return
    dirs = [WORK, ".", "/content", "/content/axialm_work", "/content/drive/MyDrive/axialm_work", "/kaggle/working"] + glob.glob("/kaggle/input/*")
    found = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "axialm_1_2_pro*.pt")):
            found[os.path.abspath(f)] = os.path.getmtime(f)
    if not found:
        print("no checkpoint found -> training starts from scratch")
        return
    best = max(found, key=found.get)
    if os.path.abspath(CKPT_PATH) != best:
        shutil.copy(best, CKPT_PATH)
        print(f"picked up {best} -> {CKPT_PATH}")
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    for key, path in (("tokenizer", TOK_PATH), ("wiki", DB_CACHE)):
        if ck.get(key) and not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(ck[key])
            print(f"restored {os.path.basename(path)} from the checkpoint")
    print(f"checkpoint: step {ck.get('step')}, {ck.get('tokens', 0) / 1e6:.0f}M tokens, RUN={ck.get('run')!r}")


def encode_parts(tok, parts):
    """assistant parts -> (ids, mask). mask = 1 where the MODEL writes the token. Tool results are injected by the
    runtime, so they get mask 0 (no loss), exactly like at inference."""
    ids, mask = [], []
    for kind, text in parts:
        t = tok.encode(text)
        if kind == "k":
            ids += [THINK] + t + [THINK_END]
            mask += [1] * (len(t) + 2)
        elif kind == "t":
            ids += t
            mask += [1] * len(t)
        elif kind == "c":
            ids += [CALL] + t + [CALL_END]
            mask += [1] * (len(t) + 2)
        else:
            ids += [RESULT] + t[:RESULT_LIMIT] + [RESULT_END]
            mask += [0] * (len(t[:RESULT_LIMIT]) + 2)
    return ids, mask


def encode_turn(tok, turn):
    role, content = turn
    if role == "user":
        t = tok.encode(content)
        return [USER] + t + [END], [0] * (len(t) + 2)
    ids, m = encode_parts(tok, content)
    return [ASSIST] + ids + [END], [0] + m + [1]


def fit_conv(tok, turns, max_len, trunc=False):
    """<bos> + as many whole (user, assistant) pairs as fit in max_len. With trunc=True a first reply that is too long
    is cut (no <eos>), which is how long UltraChat answers are used."""
    ids, mask = [BOS], [0]
    for i in range(0, len(turns) - 1, 2):
        ui, um = encode_turn(tok, turns[i])
        ai, am = encode_turn(tok, turns[i + 1])
        if len(ids) + len(ui) + len(ai) > max_len:
            room = max_len - len(ids) - len(ui)
            if i == 0 and trunc and room > 40:
                ids += ui + ai[:room]
                mask += um + am[:room]
            break
        ids += ui + ai
        mask += um + am
    return (ids, mask) if len(ids) > 1 else None


BANNED = ("as an ai", "language model developed", "openai", "chatgpt", "gpt-", "i'm sorry, but as", "as a language model", "http", "www.")


def uc_turns(msgs, max_user_words=150):
    turns = []
    for role, text in msgs:
        if not isinstance(text, str) or any(b in text.lower() for b in BANNED):
            return None
        role = "user" if role in ("user", "human") else "assistant"
        if (turns and turns[-1][0] == role) or (not turns and role != "user"):
            return None
        if role == "user":
            if len(text.split()) > max_user_words:
                return None
            turns.append(("user", text))
        else:
            turns.append(("assistant", [("k", "state: general"), ("t", text)]))
    if turns and turns[-1][0] == "user":
        turns = turns[:-1]
    return turns[:8] or None


def hf_stream(repo, name, split, seed, wid, nw, buf=5000):
    from datasets import load_dataset
    ds = load_dataset(repo, name, split=split, streaming=True) if name else load_dataset(repo, split=split, streaming=True)
    if nw > 1:
        try:
            ds = ds.shard(num_shards=nw, index=wid)
        except Exception:
            ds = ds.filter(lambda ex, i: i % nw == wid, with_indices=True)
    return ds.shuffle(seed=seed, buffer_size=buf)


CHAT_REPOS = [("stingning/ultrachat", None, "train"), ("HuggingFaceH4/ultrachat_200k", None, "train_sft")]
WEB_REPO = ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train")


def uc_messages(ex):
    if "messages" in ex:
        return [(m["role"], m["content"]) for m in ex["messages"]]
    return [("user" if i % 2 == 0 else "assistant", t) for i, t in enumerate(ex["data"])]


def check_sources():
    """Tries to read one example from each real data source; sources that fail are dropped from the blend."""
    out = dict(web=False, chat=None)
    if USE_WEB:
        try:
            next(iter(hf_stream(*WEB_REPO, 0, 0, 1, buf=10)))
            out["web"] = True
        except Exception as e:
            print("FineWeb-Edu unavailable -> web share goes to the synthetic buckets:", repr(e)[:120])
    if USE_CHAT_DATA:
        for repo in CHAT_REPOS:
            try:
                next(iter(hf_stream(*repo, 0, 0, 1, buf=10)))
                out["chat"] = repo
                break
            except Exception as e:
                print(f"{repo[0]} unavailable:", repr(e)[:100])
        if not out["chat"]:
            print("UltraChat unavailable -> chat share goes to the synthetic buckets")
    return out


def synth_docs(tok, bucket, ctx, seed, db):
    s = Synth(seed, db, make_info(CFG))
    while True:
        out = fit_conv(tok, s.doc(bucket), ctx + 1)
        if out:
            yield out


def web_docs(tok, ctx, seed, wid, nw, fallback, chunks=3):
    fails = 0
    while True:
        try:
            for ex in hf_stream(*WEB_REPO, seed + fails, wid, nw):
                ids = tok.encode(ex["text"][:6000])
                for ch in range(chunks):
                    piece = ids[ch * (ctx - 1):(ch + 1) * (ctx - 1)]
                    if len(piece) < 16:
                        break
                    yield [BOS] + piece, [0] + [1] * len(piece)
            fails = 0
        except Exception as e:
            fails += 1
            print("web stream error:", repr(e)[:100])
            time.sleep(min(60, 5 * fails))
            if fails > 8:
                yield from fallback


def chat_docs(tok, ctx, seed, wid, nw, repo, fallback):
    fails = 0
    while True:
        try:
            for ex in hf_stream(*repo, seed + fails, wid, nw):
                turns = uc_turns(uc_messages(ex))
                out = fit_conv(tok, turns, ctx + 1, trunc=True) if turns else None
                if out:
                    yield out
            fails = 0
        except Exception as e:
            fails += 1
            print("chat stream error:", repr(e)[:100])
            time.sleep(min(60, 5 * fails))
            if fails > 8:
                yield from fallback


class RowStream(IterableDataset):
    """Infinite stream of training rows (ctx+1 tokens): whole conversations packed one after another.
    seg[i] = which conversation a token belongs to (attention is blocked across them), pos[i] = position inside it."""

    def __init__(self, tok, cfg, mix, seed, db, sources):
        self.tok, self.cfg, self.seed, self.db = tok, cfg, seed, db
        self.mix = {b: w for b, w in mix.items() if w > 0}
        self.web_ok, self.chat_repo = sources["web"], sources["chat"]
        for b, ok in (("web", self.web_ok), ("chat", bool(self.chat_repo))):
            if b in self.mix and not ok:
                moved = self.mix.pop(b)
                for k in ("proc", "agent"):
                    self.mix[k] = self.mix.get(k, 0) + moved * (0.5 if k == "proc" else 0.5)
        tot = sum(self.mix.values())
        self.mix = {b: w / tot for b, w in self.mix.items()}

    def _gens(self, wid, nw):
        ctx, rs = self.cfg.ctx, self.seed * 7919 + wid * 104729
        synth = lambda b, off: synth_docs(self.tok, b, ctx, rs + off, self.db)
        g = {}
        if "proc" in self.mix:
            g["proc"] = synth("proc", 1)
        if "agent" in self.mix:
            g["agent"] = synth("agent", 2)
        if "web" in self.mix:
            g["web"] = web_docs(self.tok, ctx, rs + 3, wid, nw, synth("proc", 5))
        if "chat" in self.mix:
            g["chat"] = chat_docs(self.tok, ctx, rs + 4, wid, nw, self.chat_repo, synth("proc", 6))
        return g

    def _row(self, gen, pend):
        L = self.cfg.ctx + 1
        tk, mk = np.full(L, PAD, dtype=np.int64), np.zeros(L, dtype=np.float32)
        sg, ps = np.full(L, -1, dtype=np.int64), np.zeros(L, dtype=np.int64)
        n = k = tries = 0
        stash = []
        while L - n >= 8 and tries < 4:
            ids, m = pend.pop() if pend else next(gen)
            ln = len(ids)
            if n + ln > L:                       # does not fit here: keep it for a later row and try another one
                stash.append((ids, m))
                tries += 1
                continue
            tk[n:n + ln], mk[n:n + ln], sg[n:n + ln], ps[n:n + ln] = ids, m, k, np.arange(ln)
            n += ln
            k += 1
        pend.extend(stash)
        del pend[:max(0, len(pend) - 6)]
        return tk, mk, sg, ps

    def __iter__(self):
        wi = get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        gens = self._gens(wid, nw)
        pend = {b: [] for b in gens}
        names = list(gens)
        p = np.array([self.mix[b] for b in names])
        rng = np.random.default_rng(self.seed * 31 + wid)
        while True:
            b = names[int(rng.choice(len(names), p=p))]
            yield self._row(gens[b], pend[b])

    def rows(self, bucket, n):
        g, pend = self._gens(0, 1)[bucket], []
        return [self._row(g, pend) for _ in range(n)]


def collect_tokenizer_texts(n_web, n_chat, n_synth, db, sources):
    texts = []
    if sources["web"]:
        try:
            for i, ex in enumerate(hf_stream(*WEB_REPO, SEED, 0, 1)):
                if i >= n_web:
                    break
                texts.append(ex["text"][:3000])
        except Exception as e:
            print("web sample for the tokenizer failed:", repr(e)[:100])
    if sources["chat"]:
        try:
            for i, ex in enumerate(hf_stream(*sources["chat"], SEED, 0, 1)):
                if i >= n_chat:
                    break
                texts += [t for _, t in uc_messages(ex)][:4]
        except Exception as e:
            print("chat sample for the tokenizer failed:", repr(e)[:100])
    s = Synth(SEED, db, make_info(CFG))
    for i in range(n_synth):
        for turn in s.doc("proc" if i % 2 else "agent"):
            texts += turn_texts(turn)
    return texts


def build_tokenizer(cfg, db, sources):
    if os.path.exists(TOK_PATH):
        tok = Tokenizer.load(TOK_PATH)
        print(f"loaded tokenizer ({tok.vocab_size} tokens) from {TOK_PATH}")
    else:
        t0 = time.time()
        nw, nc, ns = RUNCFG["tok_docs"]
        tok = Tokenizer.train(collect_tokenizer_texts(nw, nc, ns, db, sources), cfg.vocab_size)
        tok.save(TOK_PATH)
        print(f"tokenizer trained: {tok.vocab_size} tokens in {time.time() - t0:.0f}s -> {TOK_PATH}")
    if tok.vocab_size != cfg.vocab_size:
        print(f"note: tokenizer has {tok.vocab_size} tokens (target {cfg.vocab_size}); the model uses {tok.vocab_size}")
        cfg.vocab_size = tok.vocab_size
    bad = scratch_collisions(tok.tokens)
    assert not bad, f"tokens Scratch would treat as equal: {bad[:5]}"
    return tok


# %% 6 - model: decoder-only, RMSNorm, RoPE, grouped-query attention (6 query heads share 2 KV heads), SwiGLU, tied embeddings
def pick_group(n, cap):
    for g in range(min(cap, n), 0, -1):
        if n % g == 0:
            return g
    return 1


def fake_quant(w, levels, cap):
    """Per-group fake quantization to `levels` values (groups run along the input dimension, one float scale per group).
    Odd levels (69): integer grid -34..34.  Even levels (8): half-integer grid -3.5..3.5, i.e. (code - 3.5) * scale with
    code 0..7 - the export stores `code`, and two codes share one character (8*8 = 64 <= 69)."""
    g = pick_group(w.shape[-1], cap)
    wg = w.float().reshape(-1, w.shape[-1] // g, g)
    half = (levels - 1) / 2
    off = 0.5 if levels % 2 == 0 else 0.0
    lo = -(levels // 2)
    scale = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / half
    q = (((wg / scale - off).round().clamp(lo, lo + levels - 1) + off) * scale).reshape(w.shape)
    return w + (q - w).detach()


class QLinear(nn.Linear):
    def __init__(self, n_in, n_out, levels, cap):
        super().__init__(n_in, n_out, bias=False)
        self.levels, self.cap, self.qat = levels, cap, False

    def forward(self, x):
        return F.linear(x, fake_quant(self.weight, self.levels, self.cap) if self.qat else self.weight)


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps) * self.w.float()).type_as(x)


def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c, s = (cos[None, None], sin[None, None]) if cos.dim() == 2 else (cos[:, None], sin[:, None])
    c, s = c.to(x.dtype), s.to(x.dtype)
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h, self.kvh, self.hd = c.n_heads, c.n_kv_heads, c.head_dim
        self.rep = self.h // self.kvh          # query head i uses kv head i // rep
        q, cap = c.quant_levels, c.quant_group
        self.wq = QLinear(c.d_model, self.h * self.hd, q, cap)
        self.wk = QLinear(c.d_model, self.kvh * self.hd, q, cap)
        self.wv = QLinear(c.d_model, self.kvh * self.hd, q, cap)
        self.wo = QLinear(self.h * self.hd, c.d_model, q, cap)

    def forward(self, x, cos, sin, mask=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.kvh, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.kvh, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k, v = k.repeat_interleave(self.rep, dim=1), v.repeat_interleave(self.rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)
        return self.wo(y.transpose(1, 2).reshape(B, T, self.h * self.hd))


class SwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate = QLinear(c.d_model, c.ffn_dim, c.quant_levels, c.quant_group)
        self.up = QLinear(c.d_model, c.ffn_dim, c.quant_levels, c.quant_group)
        self.down = QLinear(c.ffn_dim, c.d_model, c.quant_levels, c.quant_group)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.attn = RMSNorm(c.d_model, c.norm_eps), Attention(c)
        self.n2, self.ffn = RMSNorm(c.d_model, c.norm_eps), SwiGLU(c)

    def forward(self, x, cos, sin, mask=None):
        x = x + self.attn(self.n1(x), cos, sin, mask)
        return x + self.ffn(self.n2(x))


class AxiaLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.qat = cfg, False
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        hd = cfg.head_dim
        inv = 1.0 / (cfg.rope_base ** (torch.arange(0, hd, 2).float() / hd))
        ang = torch.arange(cfg.ctx).float()[:, None] * inv[None, :]
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0.0, 0.02)
        resid = 0.02 / math.sqrt(2 * cfg.n_layers)
        for b in self.blocks:
            nn.init.normal_(b.attn.wo.weight, 0.0, resid)
            nn.init.normal_(b.ffn.down.weight, 0.0, resid)

    def set_qat(self, on):
        self.qat = on
        for m in self.modules():
            if isinstance(m, QLinear):
                m.qat = on

    def forward(self, ids, seg=None, pos=None):
        T = ids.shape[1]
        E = self.emb.weight
        if self.qat:
            E = fake_quant(E, self.cfg.quant_levels, self.cfg.quant_group)
        x = F.embedding(ids, E)
        cos, sin = (self.cos[:T], self.sin[:T]) if pos is None else (self.cos[pos], self.sin[pos])
        mask = None
        if seg is not None:
            causal = torch.ones(T, T, dtype=torch.bool, device=ids.device).tril()
            mask = ((seg[:, :, None] == seg[:, None, :]) & causal)[:, None]
        for b in self.blocks:
            x = b(x, cos, sin, mask)
        return F.linear(self.norm(x), E)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# %% 7 - agent runtime (the reference the goboscript engine will copy): generate -> stop at </call> -> run tool -> inject result -> continue
def enc_assistant(tok, content):
    """History entry of an assistant turn: the exact token ids it generated (think/call/result included) or data parts."""
    if content and isinstance(content[0], int):
        return list(content)
    return encode_parts(tok, content)[0]


def build_prompt(tok, turns, budget):
    """<bos> + the most recent turns that fit in `budget` tokens (oldest whole turn pairs are dropped first) + <|assistant|>."""
    turns = list(turns)
    while True:
        ids = [BOS]
        for role, content in turns:
            if role == "user":
                ids += [USER] + tok.encode(content) + [END]
            else:
                ids += [ASSIST] + enc_assistant(tok, content) + [END]
        ids.append(ASSIST)
        if len(ids) <= budget or len(turns) <= 1:
            return ids if len(ids) <= budget else [BOS] + ids[-(budget - 1):]
        turns = turns[2:] if len(turns) > 2 else turns[1:]


def parse_reply(tok, ids):
    """Generated ids -> (think text, parts, still_open). parts = [("t", text) | ("c", call) | ("r", result)].
    The think text has no <think> / </think>; call and result markers never reach the user."""
    think, parts, buf, mode = "", [], [], "t"

    def flush():
        nonlocal think
        txt = tok.decode(buf)
        buf.clear()
        if mode == "k":
            think = txt
        elif txt.strip() or mode in ("c", "r"):
            parts.append((mode, txt))

    for t in ids:
        if t in (THINK, CALL, RESULT):
            flush()
            mode = {THINK: "k", CALL: "c", RESULT: "r"}[t]
        elif t in (THINK_END, CALL_END, RESULT_END):
            flush()
            mode = "t"
        else:
            buf.append(t)
    flush()
    return think, parts, mode != "t"


def prettify(s):
    s = re.sub(r"\bi\b", "I", s)
    return re.sub(r"(^|[.!?]\s+|\n\*?)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s)


def display_text(parts):
    """What the user sees: the visible text, with every run of tool calls collapsed into '*Called N tool(s)*'."""
    out, n = [], 0
    for kind, text in parts:
        if kind == "c":
            n += 1
        elif kind == "t":
            if n:
                out.append("*called %d tool%s*" % (n, "s" if n > 1 else ""))
                n = 0
            out.append(text.strip())
    if n:
        out.append("*called %d tool%s*" % (n, "s" if n > 1 else ""))
    return prettify("\n".join(out))


def sample_next(logits, temperature, top_k, top_p):
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k and top_k < logits.numel():
        logits[logits < torch.topk(logits, top_k).values[-1]] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        sp[(sp.cumsum(0) - sp) > top_p] = 0.0
        probs = torch.zeros_like(probs).scatter_(0, si, sp)
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1))


@torch.no_grad()
def run_agent(model, tok, history, user_msg, now=None, db=None, max_new=300, temperature=0.0, top_k=20, top_p=0.9,
              tools=True, on_update=None):
    """One assistant reply. history = [("user", text), ("assistant", ids), ...] (ids exactly as returned in out["ids"]).
    Returns dict(ids, think, parts, calls, text, display)."""
    model.eval()
    ctx = model.cfg.ctx
    ids = build_prompt(tok, list(history) + [("user", user_msg)], ctx - min(max_new, 200))
    gen, calls, call_buf = [], [], []
    in_think = in_call = False
    think_len = call_len = steps = 0
    while steps < max_new and len(ids) < ctx - 1:
        logits = model(torch.tensor([ids], device=DEVICE))[0, -1].float().clone()
        block = [PAD, BOS, USER, ASSIST, RESULT, RESULT_END]
        if gen:
            block.append(THINK)                       # <think> may only open the reply
        if not in_think:
            block.append(THINK_END)
        if not in_call:
            block.append(CALL_END)
        if in_think or in_call or not tools or len(calls) >= MAX_CALLS:
            block.append(CALL)
        if in_think or in_call:
            block.append(END)
        logits[torch.tensor(block, device=DEVICE)] = -float("inf")
        if in_think and think_len >= THINK_LIMIT:
            nxt = THINK_END
        elif in_call and call_len >= CALL_LIMIT:
            nxt = CALL_END
        else:
            nxt = sample_next(logits, temperature, top_k, top_p)
        if nxt == END:
            break
        ids.append(nxt)
        gen.append(nxt)
        steps += 1
        if nxt == THINK:
            in_think, think_len = True, 0
        elif nxt == THINK_END:
            in_think = False
        elif nxt == CALL:
            in_call, call_buf, call_len = True, [], 0
        elif nxt == CALL_END:
            in_call = False
            call_text = tok.decode(call_buf)
            res = run_tool(call_text, now, db)
            calls.append((call_text, res))
            rids = [RESULT] + tok.encode(res)[:RESULT_LIMIT] + [RESULT_END]   # injected by the runtime, not generated
            ids += rids
            gen += rids
        elif in_think:
            think_len += 1
        elif in_call:
            call_buf.append(nxt)
            call_len += 1
        if on_update:
            on_update(gen)
    think, parts, _ = parse_reply(tok, gen)
    return dict(ids=gen, think=think, parts=parts, calls=calls, steps=steps,
                text=" ".join(t for k, t in parts if k == "t"), display=display_text(parts))


def save_checkpoint(model, step, tokens_seen, opt=None, scaler=None):
    ck = {"cfg": asdict(model.cfg), "model": model.state_dict(), "step": step, "tokens": tokens_seen, "run": RUN}
    for key, path in (("tokenizer", TOK_PATH), ("wiki", DB_CACHE)):   # the .pt alone is enough to resume in a new runtime
        if os.path.exists(path):
            ck[key] = open(path, encoding="utf-8").read()
    if opt is not None:
        ck["opt"] = opt.state_dict()
    if scaler is not None and scaler.is_enabled():
        ck["scaler"] = scaler.state_dict()
    torch.save(ck, CKPT_PATH + ".tmp")
    os.replace(CKPT_PATH + ".tmp", CKPT_PATH)


def download_checkpoint():
    """No Drive: push the .pt to your computer (Colab). Upload it to the next runtime and the script picks it up by itself."""
    if USE_DRIVE:
        return
    try:
        from google.colab import files
        files.download(CKPT_PATH)
        print(f"downloading {os.path.basename(CKPT_PATH)} ({os.path.getsize(CKPT_PATH) / 1e6:.0f} MB) - keep it, it is your resume point")
    except Exception as e:
        print("could not start the download (not on Colab?):", repr(e)[:80], "- the file is", CKPT_PATH)


def load_checkpoint(path=CKPT_PATH, tok_path=TOK_PATH, quantized=True):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    model = AxiaLM(cfg)
    model.load_state_dict(ck["model"])
    model.to(DEVICE).eval()
    model.set_qat(quantized)
    return model, Tokenizer.load(tok_path)

# %% 8 - probes: end-to-end checks of tool use (real tools, greedy decoding, fixed clock)
PROBE_NOW = dt.datetime(2026, 3, 14, 15, 9, 26)   # a saturday, 3:09 pm


def make_probes(db, n=6, seed=4242):
    s = Synth(seed, db, make_info(CFG))
    P = []

    def add(kind, turns):
        P.append(dict(kind=kind, hist=turns[:-2], msg=turns[-2][1], tools=s.meta.get("tools", []), need=s.meta.get("need", [])))

    for _ in range(n):
        add("math", s.math_turn())
        add("math_chain", s.math_chain())
        add("datetime", s.time_turn(PROBE_NOW))
        add("db_seen", s.db_turn())
        add("db_heldout", s.db_turn(held=True))
        add("translate", s.translate_turn())
        add("no_tool_chat", s.chat_turn(s.c(CHAT_KEYS)))
        add("guardrail", s.guard_turn())
        add("recall", s.recall_doc(probe=True))
    return P


@torch.no_grad()
def run_probes(model, tok, probes, db, verbose=False):
    """A probe passes when the right tools (and only those) were called AND the visible answer contains the expected strings."""
    model.eval()
    res, fails = defaultdict(list), {}
    for p in probes:
        out = run_agent(model, tok, p["hist"], p["msg"], now=PROBE_NOW, db=db, max_new=200, temperature=0.0)
        called = {c.split("|")[0].strip() for c, _ in out["calls"]}
        ok = called == set(p["tools"]) and all(n in out["text"] for n in p["need"])
        res[p["kind"]].append(ok)
        if not ok and p["kind"] not in fails:
            fails[p["kind"]] = (p["msg"], out["think"], [c for c, _ in out["calls"]], out["text"], p["need"])
    model.train()
    if verbose:
        for k, (m, th, cl, tx, need) in fails.items():
            print(f"  FAIL {k}: user={m!r}\n        think={th!r} calls={cl} reply={tx!r} expected={need}")
    return {k: sum(v) / len(v) for k, v in res.items()}


def fmt_probes(res):
    return "  ".join(f"{k} {v * 100:.0f}%" for k, v in res.items()) + f"  | mean {100 * sum(res.values()) / max(1, len(res)):.0f}%"

# %% 9 - training (resumable: checkpoints hold weights + optimizer, the stream re-seeds from the saved step)
def token_weights(vocab):
    w = np.ones(vocab, dtype=np.float32)
    for t in (END, THINK, THINK_END, CALL, CALL_END):
        w[t] = SPECIAL_WEIGHT
    return torch.tensor(w, device=DEVICE)


def masked_ce(logits, y, w):
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction="none")
    w = w.reshape(-1)
    return (loss * w).sum() / w.sum().clamp(min=1.0)


def lr_at(step, total, warmup):
    if step < warmup:
        return LR * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return LR * (MIN_LR_FRAC + (1 - MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * p)))


def make_val(stream, n=16):
    val = {}
    for b in stream.mix:
        try:
            val[b] = [torch.from_numpy(np.stack(c)) for c in zip(*stream.rows(b, n))]
        except Exception as e:
            print(f"val rows for {b} unavailable:", repr(e)[:80])
    return val


@torch.no_grad()
def evaluate(model, val):
    model.eval()
    out = {}
    for b, (tk, mk, sg, ps) in val.items():
        tot, nb = 0.0, 0
        for i in range(0, tk.shape[0], 16):
            t, m, s_, p_ = (a[i:i + 16].to(DEVICE) for a in (tk, mk, sg, ps))
            with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=DEVICE.type == "cuda"):
                logits = model(t[:, :-1], s_[:, :-1], p_[:, :-1])
            tot += masked_ce(logits, t[:, 1:], m[:, 1:]).item()
            nb += 1
        out[b] = tot / nb
    model.train()
    return out


def fmt_eval(r):
    return "  ".join(f"{b} {v:.3f}" for b, v in r.items())


AMP_DTYPE = torch.bfloat16 if (DEVICE.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16   # T4 -> fp16 + GradScaler


def train(model, tok, db, sources, probes):
    cuda = DEVICE.type == "cuda"
    total = RUNCFG["tokens"] // (BATCH * CFG.ctx)
    warmup = min(1000, max(10, total // 10))
    qat_start = int(total * (1 - QAT_FRAC))
    decay = [p for p in model.parameters() if p.ndim >= 2]
    flat = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": flat, "weight_decay": 0.0}],
                            lr=LR, betas=(0.9, 0.95), eps=1e-8, fused=cuda)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda and AMP_DTYPE == torch.float16)
    step, seen = 0, 0
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False) if (RESUME and os.path.exists(CKPT_PATH)) else None
    if ck is not None and (ck.get("run") != RUN or ck.get("cfg") != asdict(CFG)):
        print(f"checkpoint {CKPT_PATH} is from a different RUN / model config (RUN={ck.get('run')!r}) - starting fresh (rename it to keep it)")
        ck = None
    if ck is not None:
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        if "scaler" in ck and scaler.is_enabled():
            scaler.load_state_dict(ck["scaler"])
        step, seen = ck["step"], ck.get("tokens", 0)
        print(f"resumed from step {step} ({seen / 1e6:.0f}M tokens)")
    if step >= total:
        print("already finished - raise RUNCFG['tokens'] (or delete the checkpoint) to train more")
        return model
    model.set_qat(step >= qat_start)
    val = make_val(RowStream(tok, CFG, MIX, SEED + 99, db, sources))
    stream = RowStream(tok, CFG, MIX, SEED + step, db, sources)
    print("blend:", {b: round(w, 3) for b, w in stream.mix.items()})
    loader = DataLoader(stream, batch_size=BATCH, num_workers=NUM_WORKERS, pin_memory=cuda,
                        prefetch_factor=4 if NUM_WORKERS else None)
    tokw = token_weights(model.cfg.vocab_size)
    fwd = torch.compile(model) if (COMPILE and cuda) else model
    print(f"training steps {step} -> {total} | {BATCH}x{CFG.ctx} tokens/step | QAT ({CFG.quant_levels} levels) from step {qat_start} "
          f"| amp {AMP_DTYPE if cuda else 'off'} | device {DEVICE}")
    model.train()
    t0, t_start, last_dl, ema, tok_run = time.time(), time.time(), time.time(), None, 0
    for batch in loader:
        if step >= total:
            break
        if step == qat_start:
            model.set_qat(True)
            print(f"--- step {step}: {CFG.quant_levels}-level weight fake-quantization is now ON ---")
        lr = lr_at(step, total, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        tk, mk, sg, ps = (a.to(DEVICE, non_blocking=True) for a in batch)
        x, y, w, seg, pos = tk[:, :-1], tk[:, 1:], mk[:, 1:] * 1.0, sg[:, :-1], ps[:, :-1]
        w = w * tokw[y]
        try:
            with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=cuda):
                loss = masked_ce(fwd(x, seg, pos), y, w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
        except Exception as e:
            if fwd is model:
                raise
            print("torch.compile failed, continuing without it:", repr(e)[:150])
            fwd = model
            continue
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()
        step += 1
        seen += x.numel()
        tok_run += x.numel()
        if step % 100 == 0 or step == 1:
            v = loss.item()
            ema = v if ema is None else 0.9 * ema + 0.1 * v
            speed = tok_run / max(1e-6, time.time() - t0)
            eta = (total - step) * x.numel() / max(1.0, speed) / 3600
            print(f"step {step:7d}/{total}  loss {v:.3f} (ema {ema:.3f})  lr {lr:.2e}  {speed / 1e3:.0f}k tok/s  eta {eta:.1f} h  "
                  f"{'qat' if model.qat else 'fp '}", flush=True)
        if step % RUNCFG["eval_every"] == 0:
            print("   val:", fmt_eval(evaluate(model, val)), flush=True)
        if step % RUNCFG["probe_every"] == 0:
            print("   probes:", fmt_probes(run_probes(model, tok, probes, db, verbose=True)), flush=True)
        if step % RUNCFG["save_every"] == 0:
            save_checkpoint(model, step, seen, opt, scaler)
        if not USE_DRIVE and (time.time() - last_dl) / 3600 >= DOWNLOAD_EVERY_HOURS:
            save_checkpoint(model, step, seen, opt, scaler)
            download_checkpoint()
            last_dl = time.time()
        if (time.time() - t_start) / 3600 > MAX_HOURS:
            save_checkpoint(model, step, seen, opt, scaler)
            download_checkpoint()
            print(f"stopped after {MAX_HOURS} h at step {step}; checkpoint saved. Next runtime: upload the .pt, run the notebook again.")
            return model
    save_checkpoint(model, step, seen, opt, scaler)
    download_checkpoint()
    model.set_qat(False)
    fp, pfp = evaluate(model, val), run_probes(model, tok, probes, db)
    model.set_qat(True)
    q, pq = evaluate(model, val), run_probes(model, tok, probes, db, verbose=True)
    print("final val  (full precision):", fmt_eval(fp), "\nfinal probes (full precision):", fmt_probes(pfp))
    print(f"final val  ({CFG.quant_levels}-level, what Scratch will run):", fmt_eval(q), "\nfinal probes (quantized):", fmt_probes(pq))
    print(f"saved {CKPT_PATH}, {TOK_PATH}, {DB_EXPORT}")
    return model

# %% 10 - chat + demo (live-updating in notebooks)
try:
    from IPython.display import display, Markdown
    HAVE_IPY = True
except Exception:
    HAVE_IPY = False


def ask(model, tok, history, msg, db=None, now=None, show=True, show_think=True, **gen):
    """Runs one agent turn and shows it. Returns the result dict; append ("assistant", out["ids"]) to the history."""
    handle = display(Markdown("*thinking...*"), display_id=True) if (HAVE_IPY and show) else None

    def render(out_ids, final=False):
        think, parts, _ = parse_reply(tok, out_ids)
        body = display_text(parts) or ("*(no reply)*" if final else "*thinking...*")
        return (("> *think: %s*\n\n" % think) if (show_think and think) else "") + body

    def live(out_ids):
        if handle is not None:
            handle.update(Markdown(render(out_ids)))

    out = run_agent(model, tok, history, msg, now=now, db=db, on_update=live, **gen)
    if handle is not None:
        handle.update(Markdown(render(out["ids"], True)))
    elif show:
        print((("[think] %s\n" % out["think"]) if show_think and out["think"] else "") + out["display"])
        for c, r in out["calls"]:
            print(f"   [tool] {c}  ->  {r}")
    return out


def chat(model, tok, db=None, temperature=0.3, top_k=20, top_p=0.9):
    history, show_think, tools, now = [], True, True, None
    print("AxiaLM 1.2 PRO chat | /reset /think /tools /temp 0.7 /time 2026-10-18T14:05:09 (fake clock) /quit")
    while True:
        try:
            msg = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            continue
        if msg == "/quit":
            break
        if msg == "/reset":
            history = []
            print("history cleared.")
        elif msg == "/think":
            show_think = not show_think
            print("show think:", show_think)
        elif msg == "/tools":
            tools = not tools
            print("tools enabled:", tools)
        elif msg.startswith("/temp"):
            try:
                temperature = float(msg.split()[1])
            except (IndexError, ValueError):
                print("usage: /temp 0.7")
        elif msg.startswith("/time"):
            try:
                now = dt.datetime.fromisoformat(msg.split()[1])
            except (IndexError, ValueError):
                now = None
            print("clock:", now or "real time")
        else:
            out = ask(model, tok, history, msg, db=db, now=now, show_think=show_think, temperature=temperature, top_k=top_k,
                      top_p=top_p, tools=tools)
            history += [("user", msg), ("assistant", out["ids"])]


DEMO = ["hello!", "who are you?", "my name is dana", "what is 43*60?", "now divide that by 4", "solve 3x + 5 = 20",
        "what time is it?", "what is the capital of japan?", "what is the capital of zorbland?", "how do you say good morning in german?",
        "what's my name?", "explain quantum string theory in depth.", "thanks!"]


def demo(model, tok, db=None, temperature=0.0):
    history = []
    for msg in DEMO:
        print(f"\nyou > {msg}")
        out = ask(model, tok, history, msg, db=db, show_think=True, temperature=temperature)
        history += [("user", msg), ("assistant", out["ids"])]


def tool_selftest():
    """No model needed: checks the tools and the display/parsing code against the plan's example (43*60)."""
    assert tool_math(["multiplication", "43", "60"]) == "2580"
    assert tool_math(["solve", "2", "3", "11"]) == "4" and tool_math(["division", "10", "3"]) == "3.3333"
    assert tool_math(["division", "1", "0"]) == "error" and tool_math(["modulo", "-7", "3"]) == "2"
    assert run_tool("datetime | date", PROBE_NOW) == "2026 | 3 | 14 | 7" and run_tool("datetime | time", PROBE_NOW) == "15 | 9 | 26"
    assert run_tool("translate | spanish | thank you") == "gracias" and run_tool("db | capital france", db=DB).startswith("the capital")
    t = Tokenizer.load(TOK_PATH) if os.path.exists(TOK_PATH) else None
    if t:
        ids, _ = encode_parts(t, [("k", "state: tool | tool: math | op: multiplication | a: 43 | b: 60"), ("t", "let me check."),
                                  ("c", "math | multiplication | 43 | 60"), ("r", "2580"), ("t", "the result of 43*60 is 2,580.")])
        th, parts, _ = parse_reply(t, ids)
        print(display_text(parts), "\n-- think:", th)
    for lv in (8, 16, 69):
        codes = [random.randrange(lv) for _ in range(1001)]
        assert unpack_codes(pack_codes(codes, lv), lv, len(codes)) == codes, lv
    assert len(pack_codes([0] * 1000, 8)) == 500       # 2 weights per character
    print("tool + packing self-test passed")


# %% 11 - build: data sources, fact database, tokenizer, model, probes
restore_from_checkpoint()   # picks up an uploaded .pt (and the tokenizer/fact list stored in it) when resuming in a new runtime
sources = check_sources()
DB = load_or_build_db()
tok = build_tokenizer(CFG, DB, sources)
model = AxiaLM(CFG).to(DEVICE)
print(f"model: {n_params(model):,} parameters ({CFG.n_layers} layers, d={CFG.d_model}, heads={CFG.n_heads}/{CFG.n_kv_heads} kv, "
      f"ffn={CFG.ffn_dim}, ctx={CFG.ctx}, vocab={CFG.vocab_size})")
PROBES = make_probes(DB)
tool_selftest()

# %% 12 - (optional) look at the training data: readable conversations, with <think> / <call> / <result> inline
# show_samples(DB, 2)

# %% 13 - train. Resumes automatically. Watch the first tok/s line; the eta column says how long the run will take.
model = train(model, tok, DB, sources, PROBES)

# %% 14 - (Colab) download checkpoint + tokenizer + fact database before the runtime is recycled
try:
    from google.colab import files
    for p in (CKPT_PATH, TOK_PATH, DB_EXPORT):
        files.download(p)
except Exception as e:
    print("not on Colab (or download blocked):", repr(e)[:100], "- the files are in", WORK)

# %% 15 - scripted demo: chat, tools (math follow-up, time, database hit + miss, translate), recall, guardrail
model, tok = load_checkpoint(quantized=True)
demo(model, tok, db=DB)

# %% 16 - talk to it (tool calls are shown as *Called N tools*; /think shows the scratchpad, /time fakes the clock)
chat(model, tok, db=DB)
