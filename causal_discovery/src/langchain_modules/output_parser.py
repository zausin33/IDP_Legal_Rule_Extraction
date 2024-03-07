from __future__ import annotations

import re

from langchain import LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BaseOutputParser, OutputParserException

FIX_PROMPT_TEMPLATE = """
Input from the user:
--------------
{user_input}
--------------

Answer:
--------------
{answer}
--------------

Instructions:
--------------
{instructions}
--------------

Above, the Answer did not satisfy the constraints given in the Instructions.
Error:
--------------
{error}
--------------

Please try again. Only answer with <Answer>ANSWER</Answer>, where ANSWER must be one of the following options: {answer_options}"""


FIX_PROMPT = PromptTemplate.from_template(FIX_PROMPT_TEMPLATE)


class AnswerOutputParser(BaseOutputParser):

    answer_options: list[str]
    regex: re.Pattern
    retry_chain: LLMChain
    with_retry_chain: bool

    @classmethod
    def from_llm(
            cls,
            llm: BaseLanguageModel,
            answer_options: list[str],
            with_retry_chain: bool = True,
    ) -> AnswerOutputParser:
        chain = LLMChain(llm=llm, prompt=FIX_PROMPT)
        concat_answer_options = "|".join(answer_options)
        pattern = f"(?i)<Answer>({concat_answer_options})</Answer>"
        regex = re.compile(pattern)
        return cls(answer_options=answer_options, regex=regex, retry_chain=chain, with_retry_chain=with_retry_chain)

    def get_format_instructions(self) -> str:
        answer_options = "/".join(self.answer_options)
        return f"Only answer with <Answer>{answer_options}</Answer>. Within the <Answer> tags, there must be on of {self.answer_options}"

    def _parse(self, text: str) -> str:
        match = re.search(self.regex, text)
        if match:
            return match.group(1)
        else:
            match = re.search(r"<Answer>(.*)</Answer>", text)
            add_error_msg = f"{match.group(1)} is not a valid answer option." if match else ""
            answer_options = "|".join(self.answer_options)
            raise OutputParserException(f"Could not find the regex \"<Answer>{answer_options}</Answer>\". {add_error_msg}")

    def parse(self, answer: str, question) -> str:
        try:
            parsed_completion = self._parse(answer)
        except OutputParserException as e:
            if self.with_retry_chain:
                new_completion = self.retry_chain.run(
                    user_input=question,
                    answer=answer,
                    instructions=self.get_format_instructions(),
                    error=repr(e),
                    answer_options=self.answer_options
                )
                print(f"New completion: {new_completion}")
                try:
                    parsed_completion = self._parse(new_completion)
                    print(f"Parsed completion: {parsed_completion}")
                except OutputParserException as e:
                    parsed_completion = "No answer found"
            else:
                parsed_completion = "No answer found"

        return parsed_completion


class ScoreOutputParser(BaseOutputParser):
    def get_format_instructions(self) -> str:
        return ""

    def parse(self, text: str) -> int:
        match = re.search(r'<Score>(\d+)</Score>', text)
        if match:
            return match.group(1)
        else:
            return -1
