import pandas as pd
from core.agent.data_agent import ask_data_agent

def main():
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Emma'],
        'department': ['Engineering', 'Engineering', 'HR', 'Sales', 'Sales'],
        'salary': [55000, 62000, 48000, 52000, 45000],
        'age': [28, 34, 29, 45, 31]
    })
    
    queries = [
        "What is the average salary by department?",
        "What is the average salary by department and how many employees are in each department?",
        "What is the average salary, total salary, and employee count by department?",
        "How does salary vary with age?",
        "What is Alice's salary?",
        "Calculate the average salary by department. Also show the oldest age by department."
    ]
    
    for q in queries:
        print("\n" + "="*80)
        print(f"QUERY: {q}")
        print("="*80)
        
        try:
            answer, chart_path, code, logs = ask_data_agent(q, df)
            
            print("\n--- ANSWER ---")
            print(answer.encode('utf-8').decode('cp1252', errors='replace'))
            
            print("\n--- CHART PATH ---")
            print(chart_path)
            
            print("\n--- LOGS ---")
            for log in logs:
                print(f"SQL: {log['sql']}")
                print(f"Status: {log['status']}")
                if log.get('error_message'):
                    print(f"Error: {log['error_message']}")
                
        except Exception as e:
            print(f"FAILED (Application Level): {e}")

if __name__ == "__main__":
    main()
