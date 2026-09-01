import os

import pandasai as pai

from pandasai.llm.groq import GroqLLM


# Create our Groq LLM
llm = GroqLLM(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-20b",
)


# Tell PandasAI to use our Groq implementation
pai.config.set({
    "llm": llm,
})


# Load the dataset
df = pai.read_csv(
    r"G:\Kavion_small\data\sample.csv"
)


# Ask PandasAI a question
result = df.chat(
    "What is the average salary?"
)


print("\n==============================")
print("KAVION GROQ + PANDASAI TEST")
print("==============================\n")

print(result)