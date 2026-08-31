import gradio as gr
import pandas as pd

from core.agent.data_agent import ask_data_agent


current_dataframe = None


def load_csv(file):

    global current_dataframe

    if file is None:
        return "❌ Please upload a CSV file."

    try:

        current_dataframe = pd.read_csv(file.name)

        rows, columns = current_dataframe.shape

        preview = current_dataframe.head(10).to_markdown(
            index=False
        )

        return f"""
## Dataset Loaded

**Rows:** {rows}

**Columns:** {columns}

### Columns

{", ".join(map(str, current_dataframe.columns))}

### Preview

{preview}
"""

    except Exception as e:

        current_dataframe = None

        return f"""
## Error

Could not load CSV:

`{str(e)}`
"""


from PIL import Image

def run_agent(question):

    if current_dataframe is None:

        return (
            "❌ Please upload a CSV first.",
            None,
            ""
        )

    if not question.strip():

        return (
            "❌ Please enter a question.",
            None,
            ""
        )

    try:

        answer, chart_path, code, logs = ask_data_agent(
            question,
            current_dataframe
        )
        
        chart_img = None
        if chart_path:
            try:
                chart_img = Image.open(chart_path)
            except Exception:
                pass

        if chart_img is not None:
            plot_update = gr.update(value=chart_img, visible=True)
        else:
            plot_update = gr.update(value=None, visible=False)

        sql_text = "\n\n".join([f"-- Query {i+1} (Status: {log['status']}, Time: {log['execution_time_ms']}ms)\n{log['sql']}" for i, log in enumerate(logs)]) if logs else "-- No SQL generated."

        return answer, plot_update, sql_text, code

    except Exception as e:

        return (
            f"""
## ❌ Analysis Error

`{str(e)}`
""",
            gr.update(value=None, visible=False),
            "-- Error generating SQL.",
            ""
        )


def create_ui():

    with gr.Blocks(
        title="Kavion Small"
    ) as app:

        gr.Markdown(
            """
# 📊 Kavion Small

### AI-Powered CSV Data Analyst

Upload a dataset and ask questions using natural language.

**Understand → Generate → Execute → Explain**
"""
        )

        with gr.Row():

            file_input = gr.File(
                label="Upload CSV",
                file_types=[".csv"]
            )

            dataset_info = gr.Markdown(
                "No dataset loaded."
            )

        load_button = gr.Button(
            "Load Dataset",
            variant="primary"
        )

        load_button.click(
            fn=load_csv,
            inputs=file_input,
            outputs=dataset_info
        )

        gr.Markdown("---")

        question = gr.Textbox(
            label="Ask Your Data",
            placeholder=(
                "Example: What is the average "
                "salary by department?"
            ),
            lines=3
        )

        ask_button = gr.Button(
            "Analyze",
            variant="primary"
        )

        answer = gr.Markdown()

        plot = gr.Image(
            label="Visualization",
            type="pil",
            visible=False
        )

        with gr.Accordion("Developer Details", open=False):
            sql_queries = gr.Code(
                label="SQL Queries Captured",
                language="sql",
                interactive=False
            )
            generated_code = gr.Code(
                label="Generated Python",
                language="python",
                interactive=False
            )

        ask_button.click(
            fn=run_agent,
            inputs=question,
            outputs=[
                answer,
                plot,
                sql_queries,
                generated_code
            ]
        )

    return app