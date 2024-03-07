from langchain.schema import BaseOutputParser
from typing import Dict, Any, List, Tuple
import re

from abc import ABC, abstractmethod
from langchain import LLMChain

from legal_reasoning.src import graph_generation
from legal_reasoning.src.code_execution import CodeExecutor
from legal_reasoning.src.utils import Config


class CorrectionModule(ABC):
    def __init__(self, config: Config, chat_chain: LLMChain, output_parser: BaseOutputParser = None):
        self.config = config
        self.chat_chain = chat_chain
        self.output_parser = output_parser

    @abstractmethod
    def is_triggered(self, text: str, **kwargs) -> (bool, Dict[str, Any]):
        pass

    @abstractmethod
    def correct(self, text: str, **kwargs) -> str:
        pass

    def __call__(self, text: str, **kwargs) -> str:
        count_different_errors = 0
        count_same_error = 0
        warning_count = 0
        old_error_msg = ""

        while True:
            if count_different_errors >= self.config.number_retries_different_errors:
                print(f"!!!Correction module {self.__class__.__name__} is stuck in a loop of different errors!!!!")
                return text

            if count_same_error >= self.config.number_retries_same_error:
                print(f"!!!Correction module {self.__class__.__name__} is stuck in a loop of the same error!!!")
                return text

            do_trigger, correct_kwargs = self.is_triggered(text, **kwargs)

            if not do_trigger:
                return text

            if correct_kwargs["error_msg"] == old_error_msg:
                count_same_error += 1
            else:
                count_same_error = 0

            old_error_msg = correct_kwargs["error_msg"]
            count_different_errors += 1

            print(f"Correction triggered {self.__class__.__name__}")
            text = self.correct(text, **correct_kwargs, **kwargs)
            if self.output_parser is not None:
                text = self.output_parser.parse(text)

            if correct_kwargs["error_lvl"] == "Warning":
                warning_count += 1
            else:
                warning_count = 0

            if warning_count > 1:
                return text


class CorrectionModuleChain:
    def __init__(self, config: Config, correction_modules: List[CorrectionModule]):
        self.config = config
        self.correction_modules = correction_modules

    def apply_corrections(self, llm_response: str) -> Tuple[str, int]:
        index = 0
        count_new_start = 0

        while index < len(self.correction_modules):
            correction_module = self.correction_modules[index]
            original_response = llm_response
            llm_response = correction_module(llm_response)

            # If the response changed after applying the correction module and there are more correction modules to apply, start again from the beginning
            if original_response != llm_response and len(self.correction_modules) > 1:
                index = 0  # Start again from the beginning
                count_new_start += 1
            else:
                index += 1  # Move to the next module

            if count_new_start > self.config.number_retries_same_error:
                print(f"!!!!!!!!Correction module chain restarted more than {self.config.number_retries_same_error} times.!!!!!!!")
                return llm_response, count_new_start

        return llm_response, count_new_start

    def __call__(self, llm_response):
        return self.apply_corrections(llm_response)


class CompilationCorrectionModule(CorrectionModule):
    code_executor: CodeExecutor

    def __init__(self, config: Config, chat_chain: LLMChain, output_parser: BaseOutputParser, code_executor: CodeExecutor):
        super().__init__(config, chat_chain, output_parser)
        self.code_executor = code_executor

    def is_triggered(self, text: str, **kwargs) -> (bool, Dict[str, Any]):
        output = self.code_executor.compile(code=text)
        if "error" in output.lower():
            output = re.sub(r"Warning:.*\n", "", output)
            return True, {"error_lvl": "Error", "error_msg": output}

        # if "warning" in output.lower():
        #    return True, {"error_lvl": "Warning", "error_msg": output}

        return False, {}

    def correct(self, text: str, **kwargs) -> str:
        if kwargs["error_lvl"] == "Error":
            correction_prompt = f"Your prolog code \n```prolog\n{text}\n```\nhas produced some Errors. Please correct the Errors.\n Errors:\n{kwargs['error_msg']}"
            if "Syntax error: Unexpected end of file" in kwargs["error_msg"]:
                correction_prompt += "\nPlease make sure that that you initialize all predicates. When you do not know how to initilize one, initizialize it with false for now and write a comment that you need to implement it later."
        else:
            correction_prompt = f"Your prolog code has produced some Warnings. You do not have to correct them, but you should take a look at them, as they might point out inaccuracies in the translation. Improve the code, or if you don't find this necessary, repeat the code as it was.\n Warnings:\n{kwargs['error_msg']}"

        return self.chat_chain(correction_prompt)["text"]


class LoopCorrectionModule(CorrectionModule):

    def is_triggered(self, text: str, **kwargs) -> (bool, Dict[str, Any]):
        G = graph_generation.read_rules(text, output_html_name="resources/tmp.html")
        loops = graph_generation.detect_loops(G)
        if len(loops) > 0:
            return True, {"loops": loops, "error_lvl": "Error",
                          "error_msg": f"Your prolog code has produced those loops {loops}."}

        return False, {}

    def correct(self, text: str, **kwargs) -> str:
        correction_prompt = (f"Your prolog code has produced some loops. Please correct the code so that he no longer "
                             f"contains any loops.\n As loops are not allowed in prolog, you probably "
                             f"misinterpreted a rule"
                             f"Predicates which form the loops:\n{kwargs['loops']}.\n"
                             f"Be cautious, that you do not create new loops while correcting the code."
                             f"It also doesn't help if you introduce a new variable if it also loops with the existing variables. ")
        return self.chat_chain(correction_prompt)["text"]
