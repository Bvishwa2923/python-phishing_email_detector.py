# python-phishing_email_detector.py
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. Create a sample dataset
# --------------------------------------------------

data = {
    "email": [
        "Congratulations! You won a prize. Click here to claim your reward.",
        "Urgent! Your bank account has been suspended. Verify your account now.",
        "You have won a $1000 gift card. Click the link to receive it.",
        "Your password will expire today. Login immediately to keep your account.",
        "Security alert! Confirm your account information using this link.",
        "You are selected for a cash reward. Send your details to receive it.",

        "Hi John, please find the meeting schedule attached.",
        "Reminder: Your project meeting is tomorrow at 10 AM.",
        "Thank you for submitting your assignment. We received it successfully.",
        "Your Amazon order has been shipped and will arrive tomorrow.",
        "The team meeting has been moved to Friday afternoon.",
        "Please review the attached project report and share your feedback."
    ],

    "label": [
        "Phishing",
        "Phishing",
        "Phishing",
        "Phishing",
        "Phishing",
        "Phishing",

        "Safe",
        "Safe",
        "Safe",
        "Safe",
        "Safe",
        "Safe"
    ]
}

df = pd.DataFrame(data)

# --------------------------------------------------
# 2. Extract additional email features
# --------------------------------------------------

def count_urls(text):
    return len(re.findall(r'https?://\S+|www\.\S+', text))

def count_suspicious_words(text):
    words = [
        "urgent", "verify", "password", "click",
        "account", "winner", "prize", "login",
        "suspended", "confirm", "reward"
    ]

    text = text.lower()
    return sum(word in text for word in words)

df["url_count"] = df["email"].apply(count_urls)
df["suspicious_word_count"] = df["email"].apply(count_suspicious_words)

print("\nDataset:")
print(df)

# --------------------------------------------------
# 3. Split dataset
# --------------------------------------------------

X = df["email"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 4. Build Machine Learning Pipeline
# --------------------------------------------------

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )),

    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])

# --------------------------------------------------
# 5. Train model
# --------------------------------------------------

model.fit(X_train, y_train)

# --------------------------------------------------
# 6. Make predictions
# --------------------------------------------------

y_pred = model.predict(X_test)

# --------------------------------------------------
# 7. Accuracy
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("PHISHING EMAIL DETECTOR")
print("==============================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

# --------------------------------------------------
# 8. Classification report
# --------------------------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# --------------------------------------------------
# 9. Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Safe", "Phishing"]
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Safe", "Phishing"],
    yticklabels=["Safe", "Phishing"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Phishing Email Detection - Confusion Matrix")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 10. Test new emails
# --------------------------------------------------

test_emails = [
    "URGENT! Your account has been suspended. Click here to verify your password.",
    "Hello, please attend the project meeting tomorrow at 11 AM.",
    "Congratulations! You won a prize. Verify your account immediately."
]

predictions = model.predict(test_emails)

print("\nNew Email Predictions:")
print("------------------------------")

for email, prediction in zip(test_emails, predictions):
    print("\nEmail:", email)
    print("Prediction:", prediction)
