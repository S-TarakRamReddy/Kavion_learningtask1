import pandas as pd
import matplotlib.pyplot as plt


def execute_analysis(code: str, df: pd.DataFrame):
    """
    Execute LLM-generated analysis code in a restricted namespace.

    The generated code receives:
        df  -> copy of the user's DataFrame
        pd  -> pandas
        plt -> matplotlib.pyplot

    The code MUST create:
        result -> the actual analytical result

    If matplotlib creates a figure, the figure is returned
    separately from the analytical result.
    """

    # --------------------------------------------------------
    # Execution namespace
    # --------------------------------------------------------

    namespace = {
        "df": df.copy(),
        "pd": pd,
        "plt": plt,
    }

    # Make sure figures from previous requests do not leak
    # into the current analysis.
    plt.close("all")

    # --------------------------------------------------------
    # Restricted builtins
    # --------------------------------------------------------

    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }

    # --------------------------------------------------------
    # Execute generated code
    # --------------------------------------------------------

    exec(
        code,
        {
            "__builtins__": safe_builtins,
        },
        namespace,
    )

    # --------------------------------------------------------
    # Retrieve analytical result
    # --------------------------------------------------------

    result = namespace.get("result")

    if result is None:
        raise ValueError(
            "Generated code did not create "
            "a `result` variable."
        )

    # --------------------------------------------------------
    # Retrieve matplotlib figure
    # --------------------------------------------------------

    figure = None

    if len(plt.get_fignums()) > 0:
        figure = plt.gcf()

    return result, figure


def format_result(result):
    """
    Convert an analytical result into displayable text.
    """

    if isinstance(result, pd.DataFrame):

        return result.to_markdown(
            index=False
        )

    if isinstance(result, pd.Series):

        return result.to_frame().to_markdown()

    if isinstance(result, (float, int)):

        return str(
            round(result, 4)
        )

    return str(result)