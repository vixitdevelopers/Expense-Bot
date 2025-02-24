import os
import sqlite3
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from transformers import pipeline, logging
from datetime import datetime

logging.set_verbosity_info()

app = Flask(__name__)

# Ensure the database is created in your project folder
DB_PATH = os.path.join(os.path.dirname(__file__), "expenses.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

# Create the expenses table (if it doesn't exist)
c.execute('''
    CREATE TABLE IF NOT EXISTS expenses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        amount REAL,
        category TEXT,
        date TEXT DEFAULT (datetime('now'))
    )
''')

# Create the categories table (if it doesn't exist)
c.execute('''
    CREATE TABLE IF NOT EXISTS categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE
    )
''')
conn.commit()

# Insert default categories if none exist (in Hebrew)
default_categories = ["אוכל", "תחבורה", "בידור", "מכולת", "חשבונות", "אחר"]
c.execute("SELECT COUNT(*) FROM categories")
if c.fetchone()[0] == 0:
    for cat in default_categories:
        c.execute("INSERT INTO categories (name) VALUES (?)", (cat,))
    conn.commit()

print('initializing classification model...')
# Initialize the zero-shot classifier using a multilingual model for better Hebrew support.
zero_shot_classifier = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-3",
    # tokenizer_kwargs={"use_fast": False}  # if needed
)
print('done init.')


def get_candidate_labels():
    """Retrieve the current list of categories from the database."""
    c.execute("SELECT name FROM categories")
    rows = c.fetchall()
    return [row[0] for row in rows]


def classify_expense(expense_name):
    """Uses NLP to determine the best category based on the expense name."""
    candidate_labels = get_candidate_labels()
    result = zero_shot_classifier(expense_name, candidate_labels)
    return result["labels"][0]


def help_message():
    return ("שלום! השתמש בפקודות הבאות:\n"
            "- הוצאה <שם> <סכום>\n"
            "- סיכום\n"
            "- רשימת קטגוריות\n"
            "- הוספת קטגוריה <שם הקטגוריה>\n"
            "- מחיקת קטגוריה <שם הקטגוריה>\n"
            "- מחיקה <שם> <סכום> (למחיקת הוצאה)")


@app.route('/whatsapp', methods=['POST'])
def whatsapp():
    incoming_msg = request.values.get('Body', '').strip()
    lower_msg = incoming_msg.lower()

    # Check if the message doesn't start with any known command.
    if not (lower_msg.startswith("הוצאה") or
            lower_msg.startswith("סיכום") or
            lower_msg.startswith("רשימת קטגוריות") or
            lower_msg.startswith("הוספת קטגוריה") or
            lower_msg.startswith("מחיקת קטגוריה") or
            lower_msg.startswith("מחיקה ")):
        # If there is no digit anywhere in the message, show the help message.
        if not any(ch.isdigit() for ch in incoming_msg):
            resp = MessagingResponse()
            msg = resp.message()
            msg.body(help_message())
            return str(resp)
        # Otherwise, default to treating it as an expense entry.
        incoming_msg = "הוצאה " + incoming_msg
        lower_msg = incoming_msg.lower()

    resp = MessagingResponse()
    msg = resp.message()

    if lower_msg.startswith('הוצאה'):
        # Expected format: הוצאה <שם> <סכום>
        parts = incoming_msg.split()
        if len(parts) < 3:
            msg.body("⚠️ נא להשתמש בפורמט: הוצאה <שם> <סכום>")
            return str(resp)
        try:
            # The last token is the amount; the rest (after "הוצאה") is the expense name.
            amount = float(parts[-1])
            name = " ".join(parts[1:-1])
            category = classify_expense(name)
            c.execute(
                'INSERT INTO expenses (name, amount, category) VALUES (?, ?, ?)',
                (name, amount, category)
            )
            conn.commit()
            msg.body(f"✅ הוצאה נרשמה:\nשם: '{name}'\nסכום: ₪{amount:.2f}\nקטגוריה: {category}")
        except Exception as e:
            msg.body("⚠️ שגיאה בעיבוד ההוצאה. ודא שהסכום הוא מספר תקין.")
    elif lower_msg.startswith("סיכום"):
        # Summarize expenses for the current month by category.
        current_year_month = datetime.now().strftime("%Y-%m")
        month_name = datetime.now().strftime("%B")
        c.execute("SELECT category, SUM(amount) FROM expenses WHERE strftime('%Y-%m', date) = ? GROUP BY category",
                  (current_year_month,))
        expenses = c.fetchall()
        if expenses:
            summary = "\n".join([f"{cat}: ₪{amt:.2f}" for cat, amt in expenses])
            msg.body(f"📊 סיכום חודשי עבור {month_name}:\n{summary}")
        else:
            msg.body(f"אין הוצאות רשומות עבור {month_name} עדיין.")
    elif lower_msg.startswith("רשימת קטגוריות"):
        # List all available categories.
        c.execute("SELECT name FROM categories")
        rows = c.fetchall()
        if rows:
            categories = [row[0] for row in rows]
            msg.body("📚 קטגוריות:\n" + "\n".join(categories))
        else:
            msg.body("לא נמצאו קטגוריות.")
    elif lower_msg.startswith("הוספת קטגוריה"):
        # Add a new category: הוספת קטגוריה <שם הקטגוריה>
        parts = incoming_msg.split()
        if len(parts) < 3:
            msg.body("⚠️ נא להשתמש בפורמט: הוספת קטגוריה <שם הקטגוריה>")
        else:
            new_category = " ".join(parts[2:])
            try:
                c.execute("INSERT INTO categories (name) VALUES (?)", (new_category,))
                conn.commit()
                msg.body(f"✅ קטגוריה '{new_category}' נוספה.")
            except sqlite3.IntegrityError:
                msg.body(f"⚠️ קטגוריה '{new_category}' כבר קיימת.")
    elif lower_msg.startswith("מחיקת קטגוריה"):
        # Delete a category: מחיקת קטגוריה <שם הקטגוריה>
        parts = incoming_msg.split()
        if len(parts) < 3:
            msg.body("⚠️ נא להשתמש בפורמט: מחיקת קטגוריה <שם הקטגוריה>")
        else:
            del_category = " ".join(parts[2:])
            c.execute("DELETE FROM categories WHERE name = ?", (del_category,))
            if c.rowcount > 0:
                conn.commit()
                msg.body(f"✅ קטגוריה '{del_category}' נמחקה.")
            else:
                msg.body(f"⚠️ קטגוריה '{del_category}' לא נמצאה.")
    elif lower_msg.startswith("מחיקה "):
        # Delete an expense: expected format: מחיקה <שם> <סכום>
        parts = incoming_msg.split()
        if len(parts) < 3:
            msg.body("⚠️ נא להשתמש בפורמט: מחיקה <שם> <סכום>")
        else:
            try:
                amount = float(parts[-1])
            except ValueError:
                msg.body("⚠️ נא לספק מספר תקין עבור הסכום.")
                return str(resp)
            name = " ".join(parts[1:-1])
            c.execute("SELECT id, date FROM expenses WHERE name=? AND amount=? ORDER BY date DESC LIMIT 1",
                      (name, amount))
            row = c.fetchone()
            if row:
                expense_id = row[0]
                c.execute("DELETE FROM expenses WHERE id=?", (expense_id,))
                conn.commit()
                msg.body(f"✅ הוצאה '{name}' בסכום ₪{amount:.2f} נמחקה.")
            else:
                msg.body(f"⚠️ לא נמצאה הוצאה תואמת בשם '{name}' עם סכום ₪{amount:.2f}.")
    else:
        msg.body(help_message())

    return str(resp)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Use Render's port or default to 10000
    app.run(host="0.0.0.0", port=port, debug=False)
