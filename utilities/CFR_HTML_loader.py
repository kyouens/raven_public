# This script processes a large regulatory HTML document, transforming it into structured data for downstream NLP tasks.
# 
# - It converts the HTML to Markdown, cleans up formatting, and extracts logical sections using regular expressions.
# - It segments the content by source sections or topics, normalizes certain heading patterns, and removes superfluous information (like 'Authority' or 'Source' notes).
# - 'Cross Reference' sections are specifically reformatted for clarity.
# - The extracted segments (pairs of source label and content) are inserted into an SQLite database for efficient querying.
# - The same data is exported as a CSV for analysis or model training.
# 
# The file expects an HTML source at sources/original/title-42.html and produces both a new SQLite database and a temporary CSV file of preprocessed data.

import html2text
import re
import sqlite3
import pandas as pd

def init_db():
    print("Initializing database...")
    conn = sqlite3.connect("./sources/SQLite/textual_regulatory_data.db")
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS regulatory_data")
    c.execute('''CREATE TABLE IF NOT EXISTS regulatory_data (Source TEXT, Content TEXT)''')
    conn.commit()
    print("Database initialized.")
    return c, conn

def insert_into_db(cursor, conn, source, content):
    print(f"Inserting data into database for source: {source}")
    cursor.execute("INSERT INTO regulatory_data (Source, Content) VALUES (?, ?)", (source, content))
    conn.commit()

def process_document(html_content):
    markdown_content = html2text.html2text(html_content)

    internal_link_pattern = r'\[([^\]]+)\]\(\/[^\)]+\)'
    cleaned_content = re.sub(internal_link_pattern, r'\1', markdown_content)

    multiline_headline_pattern = r'^(#+ [^\n]+)\n([^\n]+)\n\n'
    fixed_content = re.sub(multiline_headline_pattern, r'\1 \2\n\n', cleaned_content, flags=re.M)

    h2_text, h3_text = None, None

    def process_match(match):
        nonlocal h2_text, h3_text
        level = match.group(1)
        text = match.group(2)

        if level == "#":
            return "##### Example" if text == "Example" else ""
        elif level == "##":
            h2_text = text
        elif level == "###":
            h3_text = text
        elif level == "####":
            return f"# {text}\n_In {h2_text}_. _Topic: {h3_text}_"
        return match.group(0)

    segmented_content = re.sub(r'^(#+) ([^\n]+)', process_match, fixed_content, flags=re.M)

    segmented_content = re.sub(r'^#+\s*\n', '', segmented_content, flags=re.M)
    authority_source_pattern = r'# (Authority|Source):.*?(?=(# |\Z))'
    segmented_content = re.sub(authority_source_pattern, '', segmented_content, flags=re.S | re.M)
    cross_reference_pattern = r'(###### Cross Reference)'
    segmented_content = re.sub(cross_reference_pattern, r'> **Cross Reference**', segmented_content)
    three_lines_after_cross_ref = re.compile(r'(?<=\> \*\*Cross Reference\*\*\n)((?:.*\n){3})')
    segmented_content = re.sub(three_lines_after_cross_ref, lambda m: '> ' + '> '.join(m.group(1).splitlines(True)), segmented_content)

    segments = re.split(r'(?=^# [^\n]+\n)', segmented_content, flags=re.M)

    csv_data = [("Source", "Content")]
    for segment in segments:
        match = re.match(r'^# ([^\n]+)\n', segment, flags=re.M)
        if match:
            source = match.group(1)
            content = segment[len(match.group(0)):].strip()
            csv_data.append((source, content))

    return csv_data

def main():
    print("Starting main execution...")
    c, conn = init_db()

    print("Processing HTML file...")
    with open("sources/original/title-42.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    csv_data = process_document(html_content)

    print("Inserting data into SQLite database...")
    for source, content in csv_data[1:]:
        insert_into_db(c, conn, source, content)

    print("Closing database connection...")
    conn.close()

    print("Exporting to CSV...")
    df = pd.DataFrame(csv_data[1:], columns=csv_data[0])
    df.to_csv('./sources/temp/temporary_regulatory_data_ready.csv', index=False)

    print("Script execution completed.")

if __name__ == "__main__":
    main()