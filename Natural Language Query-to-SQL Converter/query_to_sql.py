import sqlite3
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class QueryToSQL:
    def __init__(self, db_path, model_name="t5-small"):
        """
        Initializes the Query-to-SQL converter with a database and NLP model.
        """
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def natural_to_sql(self, natural_query):
        """
        Converts natural language query into SQL using a pre-trained model.
        """
        inputs = self.tokenizer.encode("translate English to SQL: " + natural_query, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=100, num_beams=5, early_stopping=True)
        sql_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return sql_query

    def execute_query(self, natural_query):
        """
        Executes the SQL query converted from a natural language query.
        """
        sql_query = self.natural_to_sql(natural_query)
        print(f"Generated SQL: {sql_query}")
        try:
            self.cursor.execute(sql_query)
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            return f"SQL Error: {e}"

# Example Usage
# Create a SQLite database and table before running
# conn = sqlite3.connect("example.db")
# conn.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER);")
# conn.execute("INSERT INTO users VALUES (1, 'Alice', 30);")
# conn.commit()

converter = QueryToSQL("example.db")
results = converter.execute_query("Show me all users older than 25.")
print(results)
