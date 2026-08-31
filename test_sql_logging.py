import json
import os
import pandas as pd
from core.agent.data_agent import ask_data_agent

def verify_and_print_log(question: str):
    log_file = "data/sql_queries.json"
    if not os.path.exists(log_file):
        print(f"FAILED: Log file {log_file} does not exist.")
        return

    with open(log_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print("FAILED: Log file is empty.")
        return

    # Check all records for the current question
    records = [r for r in data if r.get('question') == question]
    
    if not records:
        print(f"FAILED: No records found for question '{question}'.")
        return

    print(f"Found {len(records)} SQL queries captured for this question:")
    for record in records:
        print("-" * 50)
        print(f"QUESTION:       {record['question']}")
        print(f"REQUEST ID:     {record.get('request_id')}")
        print(f"SQL QUERY:\n{record['sql']}")
        print(f"QUERY TYPE:     {record.get('query_type')}")
        print(f"STATUS:         {record['status']}")
        print(f"ROWS RETURNED:  {record.get('rows_returned')}")
        print(f"VISUALIZATION?: {record.get('visualization_requested')}")
        print(f"EXECUTION TIME: {record['execution_time_ms']} ms")
        if record.get('error'):
            print(f"ERROR MESSAGE:  {record['error']}")
        print(f"LOG FILE:       {log_file} (Record ID: {record['id']})")
    print("-" * 50)

def main():
    print("==================================================")
    print("   TESTING SQL LOGGING MECHANISM")
    print("==================================================")

    # Prepare dataset
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Emma'],
        'department': ['Engineering', 'Engineering', 'HR', 'Sales', 'Sales'],
        'salary': [55000, 62000, 48000, 52000, 45000]
    })

    tests = [
        {
            "name": "1. Simple analytical query",
            "question": "What is the total salary?",
        },
        {
            "name": "2. Grouped aggregation",
            "question": "What is the average salary by department?",
        },
        {
            "name": "3. Filtering",
            "question": "Show employees with salary greater than 50000",
        },
        {
            "name": "4. Visualization request",
            "question": "Show average salary by department as a bar chart",
        },
        {
            "name": "5. Multiple analytical questions",
            "question": "What is the average salary by department and which department has the highest average salary?",
        },
        {
            "name": "6. Multiple questions + visualization",
            "question": "What is the average salary by department, which department has the highest average salary, and show me a bar chart.",
        },
        {
            "name": "7. Failed SQL",
            # We induce failure by asking for a column that does not exist. PandasAI might generate invalid SQL or fail.
            "question": "Calculate the average of the non_existent_column column.",
        },
        {
            "name": "8. Request isolation (A)",
            "question": "What is the sum of salaries in Engineering?",
        },
        {
            "name": "8. Request isolation (B)",
            "question": "What is the sum of salaries in HR?",
        }
    ]

    for test in tests:
        print(f"\n\n[RUNNING TEST] {test['name']}")
        try:
            ask_data_agent(test['question'], df)
        except Exception as e:
            print(f"Execution Error: {e}")
        
        # Verify
        verify_and_print_log(test['question'])

if __name__ == "__main__":
    main()
