import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ── Download NLTK data (only on first run) ──────────────────────────────────
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📩",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    /* ── Global reset ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ── App background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }

    /* ── Hero header ── */
    .hero-wrapper {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #a5b4fc;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: clamp(2.2rem, 6vw, 3.4rem);
        font-weight: 800;
        background: linear-gradient(90deg, #e0e7ff 0%, #a5b4fc 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.6rem;
        line-height: 1.15;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 300;
        margin: 0;
    }

    /* ── Card wrapper ── */
    .card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2rem 2rem 1.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1.5rem;
    }

    /* ── Label above textarea ── */
    .input-label {
        color: #cbd5e1;
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    /* ── Textarea override ── */
    textarea {
        background: rgba(15, 15, 30, 0.7) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
        resize: vertical !important;
        transition: border-color 0.2s ease !important;
    }
    textarea:focus {
        border-color: rgba(99, 102, 241, 0.75) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
    }

    /* ── Predict button ── */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%);
        color: #ffffff;
        font-family: 'Syne', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        cursor: pointer;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 4px 24px rgba(99, 102, 241, 0.35);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.5);
    }
    div.stButton > button:active {
        transform: translateY(0);
    }

    /* ── Result boxes ── */
    .result-spam {
        background: linear-gradient(135deg, rgba(239,68,68,0.12) 0%, rgba(220,38,38,0.08) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-left: 4px solid #ef4444;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-top: 1.2rem;
        text-align: center;
    }
    .result-ham {
        background: linear-gradient(135deg, rgba(34,197,94,0.12) 0%, rgba(22,163,74,0.08) 100%);
        border: 1px solid rgba(34, 197, 94, 0.4);
        border-left: 4px solid #22c55e;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-top: 1.2rem;
        text-align: center;
    }
    .result-icon {
        font-size: 2.8rem;
        margin-bottom: 0.4rem;
        line-height: 1;
    }
    .result-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0.2rem 0 0.4rem;
    }
    .result-spam .result-title  { color: #fca5a5; }
    .result-ham  .result-title  { color: #86efac; }
    .result-desc {
        font-size: 0.88rem;
        font-weight: 300;
    }
    .result-spam .result-desc { color: #fca5a5cc; }
    .result-ham  .result-desc { color: #86efaccc; }

    /* ── Info pills row ── */
    .pills-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 1rem;
        justify-content: center;
    }
    .pill {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 999px;
        padding: 0.3rem 0.85rem;
        font-size: 0.75rem;
        color: #94a3b8;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }

    /* ── Divider ── */
    hr {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.07);
        margin: 1.5rem 0;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.78rem;
        padding: 1.5rem 0 2rem;
    }
    .footer a { color: #6366f1; text-decoration: none; }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load model and vectorizer ────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

try:
    model, vectorizer = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Preprocessing ────────────────────────────────────────────────────────────
ps = PorterStemmer()

def preprocess(text: str) -> str:
    """
    Full preprocessing pipeline identical to training:
      1. Lowercase
      2. Tokenise
      3. Keep alphanumeric tokens only  (removes punctuation)
      4. Remove stopwords
      5. Stem with PorterStemmer
    """
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    filtered = [
        ps.stem(word)
        for word in tokens
        if word.isalnum()
        and word not in stopwords.words("english")
        and word not in string.punctuation
    ]
    return " ".join(filtered)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-wrapper">
        <div class="hero-badge">✦ NLP · Naive Bayes · TF-IDF</div>
        <h1 class="hero-title">📩 Spam Classifier</h1>
        <p class="hero-sub">Check whether a message is Spam or Not</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Model status warning ──────────────────────────────────────────────────────
if not model_loaded:
    st.warning(
        "⚠️  **model.pkl** or **vectorizer.pkl** not found. "
        "Place both files in the same directory as `app.py` and restart the app.",
        icon="🚨",
    )
    st.stop()

# ── Input card ───────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="input-label">✉️ &nbsp; YOUR MESSAGE</p>', unsafe_allow_html=True)

user_input = st.text_area(
    label="message",
    placeholder="Enter your message here...\n\nExample: Congratulations! You've won a free iPhone. Click here to claim now.",
    height=160,
    label_visibility="collapsed",
)

predict_clicked = st.button("🔍 &nbsp; Predict", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Prediction ───────────────────────────────────────────────────────────────
if predict_clicked:
    if not user_input.strip():
        st.warning("Please enter a message before predicting.", icon="✏️")
    else:
        with st.spinner("Analysing message…"):
            processed   = preprocess(user_input)
            vectorized  = vectorizer.transform([processed])
            prediction  = model.predict(vectorized)[0]

        if prediction == 1:          # ── SPAM ──
            st.markdown(
                """
                <div class="result-spam">
                    <div class="result-icon">🚨</div>
                    <div class="result-title">Spam Message</div>
                    <div class="result-desc">
                        This message shows strong indicators of spam.<br>
                        Exercise caution — do not click any links or share personal info.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:                        # ── NOT SPAM ──
            st.markdown(
                """
                <div class="result-ham">
                    <div class="result-icon">✅</div>
                    <div class="result-title">Not Spam</div>
                    <div class="result-desc">
                        This message appears to be legitimate.<br>
                        No spam signals were detected by the classifier.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Pipeline detail pills ─────────────────────────────────────────
        st.markdown(
            f"""
            <div class="pills-row">
                <span class="pill">🔡 Lowercased</span>
                <span class="pill">✂️ Tokenised</span>
                <span class="pill">🧹 Stopwords removed</span>
                <span class="pill">🌿 Stemmed</span>
                <span class="pill">📐 TF-IDF transformed</span>
                <span class="pill">🤖 Multinomial NB</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── How it works expander ────────────────────────────────────────────────────
with st.expander("ℹ️ How does this work?"):
    st.markdown(
        """
        **Pipeline overview**

        | Step | Detail |
        |------|--------|
        | **1. Preprocessing** | Lowercase → Tokenise → Remove stopwords & punctuation → Stem (PorterStemmer) |
        | **2. Vectorisation** | TF-IDF transforms the cleaned text into a numerical feature vector |
        | **3. Classification** | Multinomial Naive Bayes predicts **Spam (1)** or **Ham (0)** |

        The model and vectorizer were serialised with `pickle` after training on the **UCI SMS Spam Collection** dataset.
        """
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <hr>
    <div class="footer">
        Built with &nbsp;<strong>Streamlit</strong> 🚀 &nbsp;·&nbsp;
        Powered by <strong>Scikit-learn</strong> &nbsp;·&nbsp;
        NLP via <strong>NLTK</strong>
    </div>
    """,
    unsafe_allow_html=True,
)