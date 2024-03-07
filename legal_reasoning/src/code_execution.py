from __future__ import annotations

import re
import subprocess
from abc import abstractmethod, ABC
from typing import TypeVar

from langchain import BasePromptTemplate, LLMChain
from langchain.output_parsers.retry import NAIVE_RETRY_WITH_ERROR_PROMPT
from langchain.schema import BaseOutputParser, OutputParserException


class CodeExecutor(ABC):

    @staticmethod
    def _prepare_code(*, code_file: str = None, code: str = None):
        if code_file is None and code is None:
            raise ValueError("Either code_file or code must be provided")
        if code_file is not None and code is not None:
            raise ValueError("Only one of code_file or code must be provided")
        if code_file is None:
            # Write code to a temporary file
            with open('temp.pl', 'w') as f:
                f.write(code)
            code_file = 'temp.pl'
        return code_file

    def execute(self, query, *, code_file: str = None, code: str = None):
        code_file = self._prepare_code(code_file=code_file, code=code)
        return self._execute(query, code_file)

    def compile(self, *, code_file: str = None, code: str = None):
        code_file = self._prepare_code(code_file=code_file, code=code)
        return self._compile(code_file)

    @abstractmethod
    def _execute(self, query, code_file: str):
        pass

    @abstractmethod
    def _compile(self, code_file: str):
        pass


class PrologExecutor(CodeExecutor):

    def _compile(self, prolog_file: str):
        cmd = ['swipl', '-s', prolog_file, '-g', 'write(\'finished compilation\')', '-t', 'halt']
        return self._execute_prolog(cmd)

    def _execute(self, query, prolog_file: str):
        cmd = ['swipl', '-s', prolog_file, '-g', query, '-t', 'halt']
        return self._execute_prolog(cmd)

    @staticmethod
    def _execute_prolog(cmd):
        # Run the Prolog code using SWI-Prolog
        try:
            # The command might vary based on your installation and OS.
            # For Windows, it might be 'swipl' instead of 'swi-prolog'
            result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=120)
            return result
        except subprocess.CalledProcessError as e:
            # This will capture the error output
            return e.output
        except subprocess.TimeoutExpired:
            return "Error prolog program timed out."


class PrologParser(BaseOutputParser):
    def get_format_instructions(self) -> str:
        return (
            "The Prolog code should be inside code blocks, like this:\n"
            "```prolog\n"
            "your code here\n"
            "```\n"
        )

    def parse(self, text: str) -> str:
        pattern = r"```(prolog)?(.*?)```"
        blocks = re.findall(pattern, text, re.DOTALL)
        if len(blocks) == 0:
            raise OutputParserException(
                "The Prolog code should be inside code blocks, like this:\n"
                "```prolog\n"
                "your code here\n"
                "```\n"
            )
        concatenated_code = "\n\n".join([block[-1] for block in blocks])
        return concatenated_code


class PrologMultipleRuleParser(BaseOutputParser):
    def get_format_instructions(self) -> str:
        return (
            "Each Rule should be inside blocks, like this:\n"
            "---\n"
            "Rule X:\n"
            "Thought: <Thought>\n"
            "Translation:\n"
            "```prolog\n"
            "<Code>\n"
            "```\n"
            "---\n"
        )

    def parse(self, text: str, *, rule_count: int) -> str:
        # remove unnecessary --- if any
        pattern = r'---\s*---'
        replacement = '---'
        text = re.sub(pattern, replacement, text)

        if not "---" in text:
            # split based on "```prolog" but keep the delimiter
            pattern = r'(```prolog)'
            text = re.sub(pattern, r'---\1', text)

        code_parts = text.split("---")
        if len(code_parts) < rule_count:
            raise OutputParserException(
                "Each Rule should be inside blocks, like this:\n"
                "---\n"
                "Rule X:\n"
                "Thought: <Thought>\n"
                "Translation:\n"
                "```prolog\n"
                "<Code>\n"
                "```\n"
                "---\n"
            )

        if len(code_parts) > rule_count:
            code_parts = code_parts[1:]
        if len(code_parts) > rule_count:
            code_parts = code_parts[:-1]
        if len(code_parts) != rule_count:
            print(f"Code parts have different length than rule count\nRule Count: {rule_count}\nCode Parts:\n{code_parts}")
        return code_parts


T = TypeVar("T")


class PrologRetryParser(BaseOutputParser[T]):
    """Wraps a parser and tries to fix parsing errors.

    Does this by passing the original prompt, the completion, AND the error
    that was raised to another language model and telling it that the completion
    did not work, and raised the given error. Differs from RetryOutputParser
    in that this implementation provides the error that was raised back to the
    LLM, which in theory should give it more information on how to fix it.
    """

    parser: BaseOutputParser[T]
    retry_chain: LLMChain
    prompt_template: BasePromptTemplate

    @classmethod
    def from_llm(
            cls,
            parser: BaseOutputParser[T],
            chain: LLMChain,
            prompt: BasePromptTemplate = NAIVE_RETRY_WITH_ERROR_PROMPT,
    ) -> PrologRetryParser[T]:
        """Create a RetryWithErrorOutputParser from an LLM.

        Args:
            llm: The LLM to use to retry the completion.
            parser: The parser to use to parse the output.
            prompt: The prompt to use to retry the completion.

        Returns:
            A RetryWithErrorOutputParser.
        """
        return cls(parser=parser, retry_chain=chain, prompt_template=prompt)

    def parse(self, completion: str, **kwargs) -> T:
        try:
            parsed_completion = self.parser.parse(completion, **kwargs)
        except OutputParserException as e:
            prompt = self.prompt_template.format(prompt="", completion=completion, error=repr(e))
            new_completion = self.retry_chain(prompt)["text"]
            parsed_completion = self.parser.parse(new_completion, **kwargs)

        return parsed_completion

    def get_format_instructions(self) -> str:
        return self.parser.get_format_instructions()

    @property
    def _type(self) -> str:
        return "retry_with_error"