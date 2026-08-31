import os
import uuid
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('Agg')

def filter_distinct_dataframes(dataframes: list) -> list:
    """Filter out duplicate or non-visualizable dataframes."""
    distinct = []
    seen_columns = []
    
    # Process in reverse to keep the FINAL dataframe when duplicates exist
    for res in reversed(dataframes):
        if isinstance(res, dict):
            df = res.get("dataframe")
            sql = res.get("sql", "")
        else:
            df = res
            sql = ""
            
        if not isinstance(df, (pd.DataFrame, pd.Series)):
            continue
        
        frame = df if isinstance(df, pd.DataFrame) else df.to_frame()
        
        # 1. Skip non-visualizable
        if frame.empty:
            continue
        num_cols = frame.select_dtypes(include='number').columns.tolist()
        if len(frame) == 1 and len(num_cols) <= 1:
            continue
            
        # 2. Check for duplicate column signatures
        col_set = frozenset(frame.columns)
        
        # We only keep one dataframe per distinct column signature.
        # This prevents redundant charts for intermediate steps (like a raw SELECT vs an aggregated GROUP BY).
        if col_set in seen_columns:
            continue
            
        distinct.append(res)
        seen_columns.append(col_set)
            
    # Reverse back to original chronological order
    distinct.reverse()
    return distinct


def extract_plot_definitions(df: pd.DataFrame) -> list:
    """Extract distinct plot definitions from a single DataFrame's analytical measures."""
    plots = []
    
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(exclude='number').columns.tolist()
    
    # 1x1 scalar ignore
    if len(df) == 1 and len(num_cols) <= 1:
        return plots
        
    # A) Multiple measures over the same categorical dimension
    if len(cat_cols) >= 1 and len(num_cols) >= 1:
        x_col = cat_cols[0]
        for y_col in num_cols:
            if pd.api.types.is_datetime64_any_dtype(df[x_col]):
                plots.append({
                    "type": "line",
                    "x": x_col,
                    "y": y_col,
                    "data": df.sort_values(by=x_col),
                    "title": f"{y_col} over {x_col}"
                })
            else:
                plots.append({
                    "type": "bar",
                    "x": x_col,
                    "y": y_col,
                    "data": df.sort_values(by=y_col, ascending=False).head(20),
                    "title": f"{y_col} by {x_col}"
                })
                
    # B) Numeric vs Numeric (no categories)
    elif len(num_cols) >= 2 and len(cat_cols) == 0:
        x_col = num_cols[0]
        y_col = num_cols[1]
        plots.append({
            "type": "scatter",
            "x": x_col,
            "y": y_col,
            "data": df,
            "title": f"{y_col} vs {x_col}"
        })
        
    # C/D) Single numeric distribution or index
    elif len(num_cols) == 1 and len(df) > 1 and len(cat_cols) == 0:
        y_col = num_cols[0]
        if len(df) > 10:
            plots.append({
                "type": "hist",
                "x": y_col,
                "data": df,
                "title": f"Distribution of {y_col}"
            })
        else:
            plots.append({
                "type": "index_bar",
                "y": y_col,
                "data": df,
                "title": f"{y_col} by index"
            })
            
    return plots


def create_subplot_visualization(dataframes: list) -> str:
    valid_results = filter_distinct_dataframes(dataframes)
    
    all_plots = []
    
    for res in valid_results:
        if isinstance(res, dict):
            df = res["dataframe"]
        else:
            df = res
            
        df = df if isinstance(df, pd.DataFrame) else df.to_frame()
        all_plots.extend(extract_plot_definitions(df))
        
    n = len(all_plots)
    if n == 0:
        return None
        
    # Determine grid
    import math
    cols = min(2, n)
    rows = math.ceil(n / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*5))
    sns.set_theme(style="whitegrid")
    
    # Flatten axes for easy iteration
    if n == 1:
        axes_list = [axes]
    elif rows == 1 or cols == 1:
        axes_list = axes if isinstance(axes, list) else axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes_list = axes.flatten()
        
    charts_drawn = 0
    
    for i, plot_def in enumerate(all_plots):
        ax = axes_list[i]
        ptype = plot_def["type"]
        df = plot_def["data"]
        
        try:
            if ptype == "line":
                sns.lineplot(data=df, x=plot_def["x"], y=plot_def["y"], marker="o", ax=ax)
                ax.set_title(plot_def["title"])
                ax.tick_params(axis='x', rotation=45)
            elif ptype == "bar":
                sns_ax = sns.barplot(data=df, x=plot_def["x"], y=plot_def["y"], hue=plot_def["x"], ax=ax)
                if sns_ax.get_legend():
                    sns_ax.get_legend().remove()
                ax.set_title(plot_def["title"])
                ax.tick_params(axis='x', rotation=45)
            elif ptype == "scatter":
                sns.scatterplot(data=df, x=plot_def["x"], y=plot_def["y"], ax=ax)
                ax.set_title(plot_def["title"])
            elif ptype == "hist":
                sns.histplot(data=df, x=plot_def["x"], kde=True, ax=ax)
                ax.set_title(plot_def["title"])
            elif ptype == "index_bar":
                sns_ax = sns.barplot(x=df.index, y=df[plot_def["y"]], hue=df.index, ax=ax)
                if sns_ax.get_legend():
                    sns_ax.get_legend().remove()
                ax.set_title(plot_def["title"])
                
            charts_drawn += 1
        except Exception as e:
            print(f"Subplot failed for a definition: {e}")
            
    # Hide any unused subplots
    for j in range(n, len(axes_list)):
        axes_list[j].set_visible(False)
        
    if charts_drawn > 0:
        plt.tight_layout()
        export_dir = os.path.join(os.getcwd(), 'exports', 'charts')
        os.makedirs(export_dir, exist_ok=True)
        chart_path = os.path.join(export_dir, f"subplot_chart_{uuid.uuid4()}.png")
        plt.savefig(chart_path)
        plt.close('all')
        return chart_path
        
    plt.close('all')
    return None

def auto_visualize(result) -> str:
    """Legacy wrapper for single result visualization."""
    return create_subplot_visualization([result])
