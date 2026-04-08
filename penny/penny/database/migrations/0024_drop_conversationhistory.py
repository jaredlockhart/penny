"""Drop conversationhistory table.

Daily/weekly topic summaries replaced by knowledge extraction and
embedding-based related message retrieval.
"""


def up(conn):
    conn.execute("DROP TABLE IF EXISTS conversationhistory")
