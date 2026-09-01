from typing import Any

from pandasai.config import Config
from pandasai.core.code_execution.environment import get_environment
from pandasai.exceptions import CodeExecutionError, NoResultFoundError


class CodeExecutor:
    """
    Handle the logic on how to handle different lines of code.
    """

    _environment: dict

    def __init__(self, config: Config) -> None:
        self._environment = get_environment()

    def add_to_env(self, key: str, value: Any) -> None:
        """
        Expose extra variables in the execution environment.
        """
        self._environment[key] = value

    def execute(self, code: str) -> dict:
        """
        Execute generated code and return the complete
        execution environment.
        """
        try:
            exec(code, self._environment)
        except Exception as e:
            raise CodeExecutionError(
                "Code execution failed"
            ) from e

        return self._environment

    def execute_and_return_result(self, code: str) -> dict:
        """
        Execute code and return both:

        1. The generated `result`
        2. The complete execution environment

        This allows downstream response handling to access
        intermediate analytical objects such as DataFrames
        generated before a chart is created.
        """

        environment = self.execute(code)

        if "result" not in environment:
            raise NoResultFoundError(
                "No result was returned from the code execution. "
                "Please return the result in dictionary format, "
                "for example: "
                "result = {'type': ..., 'value': ...}"
            )

        return {
            "result": environment["result"],
            "environment": environment,
        }

    @property
    def environment(self) -> dict:
        return self._environment