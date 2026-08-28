import pandas as pd
import pandasai as pai

from pandasai.llm.groq import GroqLLM


# ============================================================
# PandasAI + Kavion Groq LLM
# ============================================================

llm = GroqLLM(
    model="openai/gpt-oss-20b",
    temperature=0,
)

pai.config.set({
    "llm": llm,
})


# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv(
    r"G:\Kavion_small\data\sample.csv"
)

print("\n==============================")
print("DATASET")
print("==============================")

print(df)


# ============================================================
# Create PandasAI DataFrame
# ============================================================

pai_df = pai.DataFrame(df)


# ============================================================
# Ask PandasAI
# ============================================================

question = (
    "Show me the average salary by department "
    "as a bar chart"
)

print("\n==============================")
print("QUESTION")
print("==============================")

print(question)


result = pai_df.chat(question)


# ============================================================
# Inspect Response
# ============================================================

print("\n==============================")
print("RESULT TYPE")
print("==============================")

print(type(result))


print("\n==============================")
print("RESULT")
print("==============================")

print(result)


print("\n==============================")
print("RESULT DICT")
print("==============================")

if hasattr(result, "__dict__"):
    print(result.__dict__)
else:
    print("No __dict__ available.")


# ============================================================
# Generated Code
# ============================================================

print("\n==============================")
print("GENERATED CODE")
print("==============================")

if hasattr(result, "last_code_executed"):
    print(result.last_code_executed)
else:
    print("No last_code_executed attribute.")


# ============================================================
# Chart Information
# ============================================================

print("\n==============================")
print("CHART VALUE")
print("==============================")

if hasattr(result, "value"):
    print(result.value)
else:
    print("No value attribute.")


print("\n==============================")
print("CHART TYPE")
print("==============================")

if hasattr(result, "type"):
    print(result.type)
else:
    print("No type attribute.")


print("\n==============================")
print("TEST COMPLETE")
print("==============================")