import os

import pandas as pd
import pandasai as pai

from pandasai.llm.groq import GroqLLM
from pandasai.core.response.chart import ChartResponse

from core.analysis.executor import format_result


# ============================================================
# PandasAI + Groq Configuration
# ============================================================

llm = GroqLLM(
    model="openai/gpt-oss-20b",
    temperature=0,
)

import logging
logging.getLogger("pandasai").setLevel(logging.CRITICAL)

pai.config.set({
    "llm": llm,
    "save_logs": False,
    "custom_instructions": (
        "NEVER use plt.show(). "
        "ALWAYS save charts to 'exports/charts/' and return the path in the result dict. "
        "NEVER use non-ASCII characters or special hyphens in your Python comments. "
        "ALWAYS ensure the final `result` dictionary is populated."
    )
})


# ============================================================
# Result Explanation
# ============================================================

def explain_result(
    question: str,
    result,
) -> str:

    # --------------------------------------------------------
    # Convert result into text
    # --------------------------------------------------------

    if isinstance(result, pd.DataFrame):

        result_text = result.to_string(index=False)

    elif isinstance(result, pd.Series):

        result_text = result.to_string()

    elif isinstance(result, dict):

        result_text = str(result)

    else:

        result_text = str(result)

    # --------------------------------------------------------
    # Explanation prompt
    # --------------------------------------------------------

    prompt = f"""
You are a precise data analyst.

The user asked:

{question}

The analysis produced this data:

{result_text}

Explain ONLY what can be directly concluded from the
analysis result.

Rules:

1. Directly answer the user's question.
2. Mention important numerical results when available.
3. If the result contains grouped data, mention the
   important groups and their values.
4. Do not invent statistics.
5. Do not claim trends, distributions, correlations,
   causes, or patterns unless explicitly supported.
6. Do not make assumptions about the dataset.
7. Do not describe something as "balanced",
   "significant", "moderate", or "strong" unless
   supported by the actual result.
8. Keep the explanation concise.
"""

    from pandasai.core.prompts.base import BasePrompt

    class ExplanationPrompt(BasePrompt):

        def __init__(self, content: str):
            self.content = content

        def to_string(self) -> str:
            return self.content

    return llm.call(
        ExplanationPrompt(prompt),
        None,
    )


# ============================================================
# Main Data Agent
# ============================================================

def ask_data_agent(
    question: str,
    df: pd.DataFrame,
):
    """
    Run a natural-language data analysis through PandasAI.

    Returns:
        answer
        chart_path
        generated_code
    """

    # --------------------------------------------------------
    # Create PandasAI DataFrame
    # --------------------------------------------------------

    pai_df = pai.DataFrame(df)

    # --------------------------------------------------------
    # Let PandasAI perform the analysis
    # --------------------------------------------------------

    pandasai_result = pai_df.chat(
        question
    )

    # --------------------------------------------------------
    # Extract generated Python code
    # --------------------------------------------------------

    generated_code = getattr(
        pandasai_result,
        "last_code_executed",
        "",
    )

    # --------------------------------------------------------
    # Default values
    # --------------------------------------------------------

    chart_path = None

    analytical_result = pandasai_result

    # ========================================================
    # CHART RESPONSE
    # ========================================================

    if isinstance(
        pandasai_result,
        ChartResponse,
    ):

        # ----------------------------------------------------
        # Get chart path
        # ----------------------------------------------------

        chart_path = pandasai_result.value

        if chart_path and not os.path.isabs(
            chart_path
        ):
            chart_path = os.path.abspath(
                chart_path
            )

        # ----------------------------------------------------
        # Verify chart exists
        # ----------------------------------------------------

        if not chart_path or not os.path.exists(
            chart_path
        ):
            chart_path = None

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # PandasAI ChartResponse contains the underlying
        # analytical dataframe in `.data`.
        #
        # Use that dataframe for the textual explanation.
        # ----------------------------------------------------

        analytical_result = getattr(
            pandasai_result,
            "data",
            None,
        )

        # ----------------------------------------------------
        # If PandasAI did not provide data, fall back
        # gracefully.
        # ----------------------------------------------------

        if analytical_result is None:
            analytical_result = (
                "The visualization was generated, "
                "but no underlying analytical data "
                "was returned."
            )

        # ----------------------------------------------------
        # Format the actual analytical data
        # ----------------------------------------------------

        formatted_result = format_result(
            analytical_result
        )

        # ----------------------------------------------------
        # ENHANCEMENT: Extract SQL queries if present
        # This solves PandasAI's limitation of only storing
        # the LAST dataframe in ChartResponse.data
        # ----------------------------------------------------
        
        import sqlite3
        import re

        if generated_code and "SELECT" in generated_code.upper():
            try:
                conn = sqlite3.connect(':memory:')
                df.to_sql('my_table', conn, index=False)
                
                queries = set()
                for match in re.finditer(r'([\'"]{1,3})\s*(SELECT\b[\s\S]*?FROM[\s\S]*?)\1', generated_code, flags=re.IGNORECASE):
                    queries.add(match.group(2).strip())
                    
                extra_data = []
                for q in queries:
                    q = re.sub(r'(?:table_[a-z0-9_]+|\{TABLE_NAME\})', 'my_table', q, flags=re.IGNORECASE)
                    try:
                        res_df = pd.read_sql_query(q, conn)
                        extra_data.append(res_df.to_markdown(index=False))
                    except Exception:
                        pass
                
                if extra_data:
                    formatted_result = "Main Data:\n" + formatted_result + "\n\nAdditional Aggregations Extracted:\n" + "\n\n".join(extra_data)
            except Exception:
                pass


    # ========================================================
    # NORMAL DATA RESPONSE
    # ========================================================

    else:

        analytical_result = getattr(
            pandasai_result,
            "value",
            pandasai_result
        )

        formatted_result = format_result(
            analytical_result
        )

    # ========================================================
    # GENERATE EXPLANATION
    # ========================================================

    explanation = explain_result(
        question,
        formatted_result,
    )

    # ========================================================
    # BUILD ANSWER
    # ========================================================

    answer = f"""
## Answer

{explanation}

## Raw Analysis Result

{formatted_result}
"""

    # ========================================================
    # RETURN
    # ========================================================

    return (
        answer,
        chart_path,
        generated_code,
    )
    

    print("\n==============================")
    print("ASK_DATA_AGENT RETURN")
    print("==============================")
    print("ANSWER:")
    print(answer)

    print("\nCHART PATH:")
    print(chart_path)

    print("\nCHART EXISTS:")
    print(
        chart_path is not None
        and os.path.exists(chart_path)
    )

    print("\nGENERATED CODE:")
    print(generated_code)

    return (
        answer,
        chart_path,
        generated_code,
    )