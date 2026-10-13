import io
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="wide")

# ---------------------------------------------------------------------------
# Text cleaning + model training
# ---------------------------------------------------------------------------

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "to", "of", "in", "on", "for", "with", "at", "by", "from", "up",
    "about", "into", "over", "after", "you", "your", "yours", "i", "im",
    "me", "my", "we", "our", "it", "its", "this", "that", "these", "those",
    "as", "if", "so", "not", "no", "do", "does", "did", "have", "has", "had",
    "will", "would", "can", "could", "should", "just", "u", "ur", "call",
    "now", "get", "got", "please", "ok", "will", "am", "pm", "he", "she",
    "they", "them", "his", "her", "their", "what", "when", "where", "how",
    "all", "out", "up", "down", "then", "than", "there", "here", "was", "be",
}


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return text


@st.cache_resource(show_spinner="Training model on the built-in dataset...")
def train_model():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df[["v1", "v2"]].copy()
    df.columns = ["label", "message"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["label"])
    df["clean_message"] = df["message"].apply(clean_text)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["clean_message"])
    y = df["label"]

    model = MultinomialNB()
    model.fit(X, y)

    train_acc = accuracy_score(y, model.predict(X))
    return vectorizer, model, train_acc, len(df)


def predict_messages(vectorizer, model, messages: pd.Series):
    cleaned = messages.apply(clean_text)
    vec = vectorizer.transform(cleaned)
    preds = model.predict(vec)
    probs = model.predict_proba(vec)[:, 1]
    return preds, probs, cleaned


# ---------------------------------------------------------------------------
# CSV column auto-detection
# ---------------------------------------------------------------------------

MESSAGE_CANDIDATES = ["message", "text", "sms", "content", "body", "v2", "email", "msg", "email_text"]
LABEL_CANDIDATES = ["label", "class", "category", "target", "v1", "is_spam", "spam"]

SPAM_TOKENS = {"spam", "1", "yes", "true", "s"}
HAM_TOKENS = {"ham", "0", "no", "false", "not spam", "notspam", "n", "legit", "legitimate"}


def find_column(columns, candidates):
    lower_map = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def normalize_label(value):
    val = str(value).strip().lower()
    if val in SPAM_TOKENS:
        return 1
    if val in HAM_TOKENS:
        return 0
    return None


def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read()
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise ValueError("Could not read this CSV file. Please check the file format.")


def top_keywords(messages: pd.Series, top_n: int = 15):
    counter = Counter()
    for msg in messages:
        for word in msg.split():
            if len(word) > 2 and word not in STOPWORDS:
                counter[word] += 1
    return counter.most_common(top_n)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("📧 Spam Classifier")
st.caption("Check single messages, or upload a CSV of messages to get a full spam report.")

vectorizer, model, train_acc, train_size = train_model()

tab_single, tab_batch = st.tabs(["✉️ Check a single message", "📂 Batch CSV analysis"])

# ---- Tab 1: single message ----
with tab_single:
    st.subheader("Check a single message")
    message = st.text_area("Enter your message", height=120)

    if st.button("Check", type="primary"):
        if message.strip() != "":
            pred, prob, _ = predict_messages(vectorizer, model, pd.Series([message]))
            label = "🚨 Spam" if pred[0] == 1 else "✅ Not Spam"
            if pred[0] == 1:
                st.error(f"Result: {label}  (confidence: {prob[0]*100:.1f}%)")
            else:
                st.success(f"Result: {label}  (confidence: {(1-prob[0])*100:.1f}%)")
        else:
            st.warning("Please enter a message")

    with st.expander("Model info"):
        st.write(
            f"Trained on **{train_size}** labeled messages from the built-in dataset. "
            f"Training accuracy: **{train_acc*100:.1f}%**."
        )

# ---- Tab 2: batch CSV ----
with tab_batch:
    st.subheader("Upload a CSV to analyze")
    st.write(
        "Upload a CSV containing a column of messages. If it also has a label column "
        "(spam/ham), the report will include accuracy metrics."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = load_uploaded_csv(uploaded_file)
        except ValueError as e:
            st.error(str(e))
            df = None

        if df is not None and not df.empty:
            st.write(f"Loaded **{len(df)}** rows with columns: {list(df.columns)}")

            # --- column detection ---
            detected_msg_col = find_column(df.columns, MESSAGE_CANDIDATES)
            detected_label_col = find_column(df.columns, LABEL_CANDIDATES)

            col1, col2 = st.columns(2)
            with col1:
                msg_col = st.selectbox(
                    "Message column",
                    options=list(df.columns),
                    index=list(df.columns).index(detected_msg_col) if detected_msg_col else 0,
                )
            with col2:
                label_options = ["(none)"] + list(df.columns)
                default_label_idx = (
                    label_options.index(detected_label_col) if detected_label_col else 0
                )
                label_col_choice = st.selectbox("Label column (optional)", options=label_options, index=default_label_idx)
                label_col = None if label_col_choice == "(none)" else label_col_choice

            run = st.button("Analyze CSV", type="primary")

            if run:
                work_df = df.copy()
                work_df = work_df.dropna(subset=[msg_col]).reset_index(drop=True)

                with st.spinner("Classifying messages..."):
                    preds, probs, cleaned = predict_messages(vectorizer, model, work_df[msg_col])

                work_df["predicted_label"] = ["spam" if p == 1 else "ham" for p in preds]
                work_df["spam_probability"] = probs.round(4)

                has_labels = False
                if label_col is not None:
                    normalized = work_df[label_col].apply(normalize_label)
                    valid_mask = normalized.notna()
                    if valid_mask.sum() >= max(5, int(0.5 * len(work_df))):
                        has_labels = True
                        work_df["true_label_normalized"] = normalized

                total = len(work_df)
                spam_count = int((work_df["predicted_label"] == "spam").sum())
                ham_count = total - spam_count
                spam_pct = spam_count / total * 100 if total else 0
                ham_pct = 100 - spam_pct

                st.markdown("## 📊 Report")

                # --- summary metrics ---
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total messages", total)
                m2.metric("Predicted spam", f"{spam_count} ({spam_pct:.1f}%)")
                m3.metric("Predicted ham", f"{ham_count} ({ham_pct:.1f}%)")
                if has_labels:
                    y_true = work_df.loc[valid_mask, "true_label_normalized"].astype(int)
                    y_pred = pd.Series(preds)[valid_mask.values].astype(int)
                    acc = accuracy_score(y_true, y_pred)
                    m4.metric("Accuracy vs true labels", f"{acc*100:.1f}%")
                else:
                    m4.metric("Accuracy", "N/A (no labels)")

                # --- charts ---
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Spam vs Ham distribution**")
                    fig, ax = plt.subplots()
                    ax.pie(
                        [spam_count, ham_count],
                        labels=["Spam", "Ham"],
                        autopct="%1.1f%%",
                        colors=["#e74c3c", "#2ecc71"],
                        startangle=90,
                    )
                    ax.axis("equal")
                    st.pyplot(fig)

                with c2:
                    st.markdown("**Top keywords in messages flagged as spam**")
                    spam_cleaned = cleaned[pd.Series(preds) == 1]
                    keywords = top_keywords(spam_cleaned, top_n=15)
                    if keywords:
                        words, counts = zip(*keywords)
                        fig2, ax2 = plt.subplots()
                        ax2.barh(words[::-1], counts[::-1], color="#e67e22")
                        ax2.set_xlabel("Frequency")
                        st.pyplot(fig2)
                    else:
                        st.info("No spam messages detected to extract keywords from.")

                # --- accuracy details if labels present ---
                if has_labels:
                    st.markdown("**Model performance vs. provided labels**")
                    prec = precision_score(y_true, y_pred, zero_division=0)
                    rec = recall_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    pm1, pm2, pm3 = st.columns(3)
                    pm1.metric("Precision", f"{prec*100:.1f}%")
                    pm2.metric("Recall", f"{rec*100:.1f}%")
                    pm3.metric("F1 score", f"{f1*100:.1f}%")

                    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                    fig3, ax3 = plt.subplots()
                    ax3.imshow(cm, cmap="Blues")
                    ax3.set_xticks([0, 1])
                    ax3.set_yticks([0, 1])
                    ax3.set_xticklabels(["Ham", "Spam"])
                    ax3.set_yticklabels(["Ham", "Spam"])
                    ax3.set_xlabel("Predicted")
                    ax3.set_ylabel("Actual")
                    for i in range(2):
                        for j in range(2):
                            ax3.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
                    ax3.set_title("Confusion Matrix")
                    st.pyplot(fig3)
                elif label_col is not None:
                    st.info(
                        "The selected label column couldn't be reliably matched to spam/ham "
                        "values, so accuracy metrics were skipped."
                    )

                # --- results table + download ---
                st.markdown("**Detailed results**")
                st.dataframe(work_df, use_container_width=True)

                csv_bytes = work_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download results as CSV",
                    data=csv_bytes,
                    file_name="spam_classification_report.csv",
                    mime="text/csv",
                )
        elif df is not None:
            st.warning("The uploaded CSV appears to be empty.")
