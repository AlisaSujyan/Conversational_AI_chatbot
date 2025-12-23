import sqlite3

db_path = r"C:\Users\MSI GF66\PycharmProjects\Conversational_AI_chatbot\assistant_v14.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [t[0] for t in cursor.fetchall()]

print("\n📌 Tables in database:")
for t in tables:
    print(" -", t)

if not tables:
    print("\n❌ No tables found.")
    conn.close()
    exit()

print("\n==============================")
print("📌 Showing all tables with first 10 rows")
print("==============================\n")

for table in tables:
    print(f"▶ TABLE: {table}")

    # Get column names
    cursor.execute(f"PRAGMA table_info({table});")
    columns = [col[1] for col in cursor.fetchall()]
    print("   Columns:", columns)

    # Get rows
    cursor.execute(f"SELECT * FROM {table} LIMIT 10;")
    rows = cursor.fetchall()

    if rows:
        for row in rows:
            print("   ", row)
    else:
        print("   (No data)")

    print("\n------------------------------\n")

conn.close()
