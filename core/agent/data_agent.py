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

    import time
    import contextlib
    import uuid
    from pandasai.agent.base import Agent
    from core.analysis.sql_logger import SqlLogger

    sql_logger = SqlLogger()

    @contextlib.contextmanager
    def patch_agent_execute_sql(current_question: str, req_id: str, logs: list, captured: list):
        original_execute = Agent._execute_sql_query
        
        query_counter = {"index": 0}

        def patched_execute(self, query: str):
            query_counter["index"] += 1
            current_index = query_counter["index"]
            start_time = time.perf_counter()
            status = "success"
            error_message = None
            rows_returned = None
            try:
                res = original_execute(self, query)
                if isinstance(res, pd.DataFrame):
                    rows_returned = len(res)
                    captured.append({
                        "dataframe": res,
                        "sql": query,
                        "request_id": req_id,
                        "query_index": current_index
                    })
                return res
            except Exception as e:
                status = "failed"
                error_message = str(e)
                raise
            finally:
                end_time = time.perf_counter()
                execution_time_ms = int((end_time - start_time) * 1000)
                logs.append({
                    "request_id": req_id,
                    "question": current_question,
                    "sql": query,
                    "status": status,
                    "execution_time_ms": execution_time_ms,
                    "rows_returned": rows_returned,
                    "error_message": error_message
                })

        Agent._execute_sql_query = patched_execute
        try:
            yield
        finally:
            Agent._execute_sql_query = original_execute

    # --------------------------------------------------------
    # Let PandasAI perform the analysis
    # --------------------------------------------------------

    request_id = str(uuid.uuid4())
    pending_logs = []
    captured_dfs = []
    visualization_requested = False

    try:
        with patch_agent_execute_sql(question, request_id, pending_logs, captured_dfs):
            pandasai_result = pai_df.chat(question)
        
        if isinstance(pandasai_result, ChartResponse):
            visualization_requested = True
    finally:
        for log_entry in pending_logs:
            sql_logger.log_query(
                request_id=log_entry["request_id"],
                question=log_entry["question"],
                sql=log_entry["sql"],
                status=log_entry["status"],
                execution_time_ms=log_entry["execution_time_ms"],
                rows_returned=log_entry["rows_returned"],
                visualization_requested=visualization_requested,
                error_message=log_entry["error_message"]
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

        from core.analysis.visualizer import auto_visualize
        chart_path = auto_visualize(analytical_result)

    # ========================================================
    # ENHANCEMENT: Append Captured DFs
    # ========================================================
    # Ensure the LLM sees ALL underlying data fetched from SQL,
    # and auto-visualize if we don't have a chart yet.
    
    if captured_dfs:
        df_strings = []
        for c_res in captured_dfs:
            cdf = c_res["dataframe"] if isinstance(c_res, dict) else c_res
            if not cdf.empty:
                prefix = ""
                if isinstance(c_res, dict) and "query_index" in c_res:
                    prefix = f"Query #{c_res['query_index']} Result:\n"
                df_strings.append(prefix + cdf.to_markdown(index=False))
        
        if df_strings:
            formatted_result = f"Main Result:\n{formatted_result}\n\nUnderlying Executed Data:\n" + "\n\n".join(df_strings)
            
        from core.analysis.visualizer import create_subplot_visualization
        subplot_path = create_subplot_visualization(captured_dfs)
        if subplot_path:
            chart_path = subplot_path

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
        pending_logs
    )

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