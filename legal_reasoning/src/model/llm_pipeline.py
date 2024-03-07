import ast
from typing import Any, Dict
from typing import List

import networkx as nx
import numpy as np
import pandas as pd
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser

from legal_reasoning.src import prolog_utils
from legal_reasoning.src.model import prompts
from legal_reasoning.src.model.correction_module import CorrectionModuleChain
from legal_reasoning.src.model.examples import translation_examples
from legal_reasoning.src.utils import Config


########################################################################################################################
# General Pipeline Modules
########################################################################################################################

class Handler:
    def execute(self, context: dict) -> dict:
        raise NotImplementedError


class ParseTextWithLLM(Handler):
    def __init__(self, config: Config, translation_chain: LLMChain, translation_prompt: callable,
                 memory: BaseChatMemory):
        self.config = config
        self.translation_chain = translation_chain
        self.translation_prompt = translation_prompt
        self.memory = memory

    def execute(self, context: dict) -> dict:
        self.memory.clear()
        llm_input = self.translation_prompt(context["text"], "")
        llm_response = self.translation_chain(llm_input)
        llm_response = llm_response["text"]
        context["text"] = llm_response

        i = 1
        while f"parsed_text_{i}" in context:
            i += 1
        context[f"parsed_text_{i}"] = llm_response

        return context


class PrepareAndExecuteTranslationChain(Handler):
    def __init__(self, config: Config, translation_chain: LLMChain, memory: BaseChatMemory,
                 translation_prompt: callable = None):
        self.config = config
        self.translation_chain = translation_chain
        self.translation_prompt = translation_prompt
        self.memory = memory

    def execute(self, context: dict) -> dict:
        translation_prompt = context[
            "translation_prompt"] if "translation_prompt" in context else self.translation_prompt

        llm_input = translation_prompt(context["text"], context["user_edit"])
        if "llm_inputs" in context:
            context["llm_inputs"].append(llm_input)
        else:
            context["llm_inputs"] = [llm_input]
        if "llm_intermediate_outputs" not in context:
            context["llm_intermediate_outputs"] = []

        self.__clear_memory_and_set_examples(
            example_template=lambda input: translation_prompt(input, ""),
            examples=translation_examples if self.config.use_examples_for_translation else [],
            additional_input=[],  # context["llm_inputs"],
            additional_output=[]  # context["llm_intermediate_outputs"]
        )

        llm_response = self.translation_chain(llm_input)
        llm_response = llm_response["text"]
        print(llm_response)
        context["llm_response"] = llm_response

        return context

    def __clear_memory_and_set_examples(
            self,
            example_template: callable,
            examples: List[dict],
            additional_input: List[str] = None,
            additional_output: List[str] = None
    ):
        self.memory.clear()
        for example in examples:
            question = example_template(example["input"])
            output = example["output"]
            self.memory.save_context({"question": question}, {"text": output})

        if additional_input is not None and additional_output is not None:
            for input, output in zip(additional_input, additional_output):
                self.memory.save_context({"question": input}, {"text": output})


class MemorySaveContext(Handler):
    def __init__(self, memory: BaseChatMemory):
        self.memory = memory

    def execute(self, context: dict) -> dict:
        self.memory.save_context({"question": context["llm_input"]}, {"text": context["llm_response"]})
        return context


class SplitLLMResponseIntoCodeAndQuery(Handler):

    def execute(self, context: dict, default_query: str = "") -> dict:
        llm_response = context["llm_response"]

        try:
            query = llm_response.split("Query:\n```prolog\n")[1].split("\n```")[0].replace("?-", "")
            code = llm_response.split("Query:\n```prolog\n")[0]
        except IndexError as e:
            print(e)
            print("No query found for")
            print(llm_response)
            query = default_query
            code = llm_response

        context["code"] = code
        context["query"] = query
        return context


class Split(Handler):

    def __init__(self, parser):
        self.parser = parser

    def execute(self, context: dict, default_query: str = "") -> dict:
        llm_response = context["llm_response"]
        rule_count = len(context["text"].split("\n---\n")) - 1
        code_parts = self.parser.parse(llm_response, rule_count=rule_count)
        context["code"] = code_parts
        context["query"] = ""
        return context


class PrologRegexCorrection(Handler):

    def execute(self, context: dict) -> dict:
        code = context["code"]
        query = context["query"]
        if isinstance(code, list):
            for idx, c in enumerate(code):
                c = prolog_utils.replace_not_with_not_plus(c)
                code[idx] = c
        else:
            code = prolog_utils.replace_not_with_not_plus(code)

        query = prolog_utils.replace_not_with_not_plus(query)
        context["code"] = code
        context["query"] = query
        return context


class ParseAndCorrectCode(Handler):
    def __init__(self, output_parser: BaseOutputParser, llm_correction_modules: CorrectionModuleChain):
        self.output_parser = output_parser
        self.llm_correction_modules = llm_correction_modules

    def execute(self, context: dict) -> dict:
        code = context["code"]
        if isinstance(code, list):
            parsed_code = []
            correction_count = []
            for c in code:
                c = self.output_parser.parse(c)
                c, corr_count = self.llm_correction_modules(c)
                parsed_code.append(c)
                correction_count.append(corr_count)
        else:
            parsed_code = self.output_parser.parse(code)
            parsed_code, correction_count = self.llm_correction_modules(parsed_code)

        context["parsed_code"] = parsed_code
        context["correction_count"] = correction_count
        return context


class Pipeline:
    def __init__(self, tasks: List[Handler]):
        self.tasks = tasks

    def run(self, context: dict) -> dict:
        for task in self.tasks:
            context = task.execute(context)
        return context


########################################################################################################################
# GOZ specific Pipeline Modules
########################################################################################################################

class GozTextSectionHandler(Handler):

    def __init__(self, config: Config, text_section_pipline: Pipeline):
        self.config = config
        self.text_section_pipline = text_section_pipline

    def execute(self, context: dict) -> dict:
        text_df = context["text_df"]

        sections = text_df["Section"].unique()

        for section in sections:
            df = text_df[text_df["Section"] == section]
            if not len(df):
                continue
            paragraph = f""
            for idx, row in df.iterrows():
                subsection = str(row["Subsection"])
                if subsection == "nan":
                    subsection = ""
                else:
                    subsection = f"{subsection} "
                paragraph += row["rule"] + "\n"

            with get_openai_callback() as cb:
                try:
                    result = self.text_section_pipline.run({
                        "text": paragraph,
                        "user_edit": ""
                    })
                    llm_response = result["llm_response"]
                    parsed_code = result["parsed_code"]
                    query = result["query"]
                except Exception as e:
                    llm_response = e
                    parsed_code = "Error"
                    query = "Error"
                    print(e)

                index = df.index[0]
                text_df.at[index, "llm_response"] = llm_response
                text_df.at[index, "parsed_code"] = parsed_code
                text_df.at[index, "query"] = query
                i = 1
                while f"parsed_text_{i}" in context:
                    text_df.at[index, f"parsed_text_{i}"] = context[f"parsed_text_{i}"]
                    i += 1
                text_df.at[index, "llm_calls"] = cb.successful_requests
                text_df.at[index, "prompt_tokens"] = cb.prompt_tokens
                text_df.at[index, "completion_tokens"] = cb.completion_tokens
                text_df.at[index, "total_cost"] = cb.total_cost

        return context


class GozTextSectionHandlerV2(Handler):

    def __init__(self, config: Config, text_section_pipline: Pipeline):
        self.config = config
        self.text_section_pipline = text_section_pipline

    def execute(self, context: dict) -> dict:
        text_df = context["text_df"]

        for index, row in text_df.iterrows():
            with get_openai_callback() as cb:
                try:
                    result = self.text_section_pipline.run({
                        "text": row["rule"],
                        "user_edit": ""
                    })
                    llm_response = result["llm_response"]
                    parsed_code = result["parsed_code"]
                    query = result["query"]
                except Exception as e:
                    llm_response = e
                    parsed_code = "Error"
                    query = "Error"
                    print(e)

                text_df.at[index, "llm_response"] = llm_response
                text_df.at[index, "parsed_code"] = parsed_code
                text_df.at[index, "query"] = query
                i = 1
                while f"parsed_text_{i}" in context:
                    text_df.at[index, f"parsed_text_{i}"] = context[f"parsed_text_{i}"]
                    i += 1
                text_df.at[index, "llm_calls"] = cb.successful_requests
                text_df.at[index, "prompt_tokens"] = cb.prompt_tokens
                text_df.at[index, "completion_tokens"] = cb.completion_tokens
                text_df.at[index, "total_cost"] = cb.total_cost

        return context


class ConstructOneRootNode(Handler):

    def __init__(self, config: Config, translation_chain: LLMChain, memory: BaseChatMemory,
                 parse_and_correct_code: Handler):
        self.config = config
        self.translation_chain = translation_chain
        self.memory = memory
        self.parse_and_correct_code = parse_and_correct_code

    def execute(self, context: dict) -> dict:
        parsed_code = context["parsed_code"]
        root_nodes = prolog_utils.find_root_predicates(parsed_code)

        translation_prompt = lambda text, user_edit: (PromptTemplate.from_template(prompts.CONSTRUCT_ONE_ROOT_PROMPT)
                                                      .format(code=parsed_code, predicates=str(root_nodes)))

        parse_text_with_llm_handler = ParseTextWithLLM(self.config, self.translation_chain, translation_prompt,
                                                       self.memory)
        parse_text_with_llm_handler.execute(context)
        new_code = context["text"]
        context["code"] = new_code
        context = self.parse_and_correct_code.execute(context)
        new_parsed_code = context["parsed_code"]
        code = parsed_code + "\n\n" + new_parsed_code
        context["parsed_code"] = code
        return context


class BaseSectionHandler(Handler):
    def __init__(self, config: Any, text_section_pipeline: Any):
        self.config = config
        self.text_section_pipeline = text_section_pipeline

    def process_sections(self, context: Dict[str, Any], section_df: pd.DataFrame, sections: List[str],
                         section_column_name: str, translated_sections: Dict[str, str]):
        for section in sections:
            df = section_df[section_df[section_column_name] == section]
            self.process_section(section_df, df, context, translated_sections, current_section=section)

    def process_section(self, section_df: pd.DataFrame, df: pd.DataFrame, context: Dict[str, Any],
                        translated_sections: Dict[str, str], current_section: str):
        # filter rule must not be empty
        rule_column_name = "rule" if self.config.extract_rules_first else "content"
        df = df[df[rule_column_name].apply(lambda x: x is not None and str(x) != "nan" and len(x.strip()) > 0)]

        if not len(df):
            return

        if len(df) > 15:
            # split df in array of dfs, where each df is no longer then 15
            dfs = np.array_split(df, len(df) // 15 + 1)
        else:
            dfs = [df]

        for df in dfs:
            paragraph, ref_sections = self.prepare_paragraph_and_references(df, current_section, rule_column_name)
            commentary = "<<".join(set(df["commentary"].values)) if self.config.use_goz_commentary and "commentary" in df else ""
            prompt = self.create_prompt(ref_sections, translated_sections, commentary)

            with get_openai_callback() as cb:
                try:
                    llm_inputs, llm_response, parsed_code, query, correction_counts = self.execute_pipeline(paragraph, prompt, current_section,
                                                                                         translated_sections)
                    self.update_dataframe(cb, context, df, llm_inputs, llm_response, parsed_code, query, section_df, correction_counts)
                except Exception as e:
                    print("Error occurred in section", current_section)
                    print(e)
                    print(prompt)
                    print(df)

    @staticmethod
    def prepare_paragraph_and_references(df: pd.DataFrame, current_section: str, rule_column_name: str) -> (str, list):
        paragraph = ""
        ref_sections = []
        for idx, row in df.iterrows():
            paragraph += row[rule_column_name] + "\n---\n"
            references = ast.literal_eval(row["parsed_references"]) if row["parsed_references"] and str(
                row["parsed_references"]) != "nan" else []
            for ref in references:
                if "section" in ref and ref["section"]:
                    ref_sections.append(ref["section"])

        ref_sections = set(ref_sections) - {str(current_section)}
        return paragraph, ref_sections

    @staticmethod
    def create_prompt(ref_sections: list, translated_sections: dict, commentary: str):
        if len(ref_sections):
            translated_references = "\n\n".join(
                [f"§ {ref_section_number}:\n{translated_sections[ref_section_number]}" for ref_section_number in
                 ref_sections if ref_section_number in translated_sections])
            if commentary:
                prompt_template = PromptTemplate.from_template(
                    prompts.PROLOG_TRANSLATION_PROMPT_WITH_REFERENCE_AND_COMMENTARY)

                prompt = lambda text, user_edit: prompt_template.format(
                    text=text, rule_count=len(text.split("\n---\n")) - 1, references=translated_references, commentary=commentary
                )
            else:
                prompt_template = PromptTemplate.from_template(
                    prompts.PROLOG_TRANSLATION_PROMPT_WITH_REFERENCE)

                prompt = lambda text, user_edit: prompt_template.format(
                    text=text, rule_count=len(text.split("\n---\n")) - 1, references=translated_references
                )
        elif commentary:
            prompt_template = PromptTemplate.from_template(prompts.PROLOG_TRANSLATION_PROMPT_WITH_COMMENTARY)
            prompt = lambda text, user_edit: prompt_template.format(text=text, rule_count=len(text.split("\n---\n")) - 1, commentary=commentary)
        else:
            prompt_template = PromptTemplate.from_template(prompts.PROLOG_TRANSLATION_PROMPT)
            prompt = lambda text, user_edit: prompt_template.format(text=text, rule_count=len(text.split("\n---\n")) - 1)
        return prompt

    def execute_pipeline(self, paragraph, prompt, section_number, translated_sections):
        try:
            result = self.text_section_pipeline.run({
                "text": paragraph,
                "user_edit": "",
                "translation_prompt": prompt,
            })
            llm_response = result["llm_response"]
            parsed_code = result["parsed_code"]
            self.update_translated_sections(translated_sections, section_number, parsed_code)
            query = result["query"]
            llm_inputs = result["llm_inputs"]
            correction_counts = result["correction_count"]
        except Exception as e:
            llm_response = e
            parsed_code = "Error"
            query = "Error"
            llm_inputs = []
            correction_counts = 0
            print(e)
        return llm_inputs, llm_response, parsed_code, query, correction_counts

    @staticmethod
    def update_dataframe(cb, context, df, llm_inputs, llm_response, parsed_code, query, section_df, correction_counts):
        index = df.index[0]
        section_df.at[index, "llm_response"] = llm_response
        section_df.at[index, "query"] = query
        section_df.at[index, "llm_inputs"] = llm_inputs[0]
        i = 1
        while f"parsed_text_{i}" in context:
            section_df.at[index, f"parsed_text_{i}"] = context[f"parsed_text_{i}"]
            i += 1
        section_df.at[index, "llm_calls"] = cb.successful_requests
        section_df.at[index, "prompt_tokens"] = cb.prompt_tokens
        section_df.at[index, "completion_tokens"] = cb.completion_tokens
        section_df.at[index, "total_cost"] = cb.total_cost

        for i, (idx, row) in enumerate(df.iterrows()):
            section_df.at[idx, "parsed_code"] = parsed_code[i]
            section_df.at[idx, "correction_count"] = correction_counts[i]

    def update_translated_sections(self, translated_sections, section_number, parsed_code):
        pass


class GozTextSectionHandlerDependencyTree(BaseSectionHandler):
    def execute(self, context: dict) -> dict:
        text_df = context["text_df"]
        G = self.construct_dependency_graph(text_df, only_sections=True)
        section_order = self.postorder_traversal_forest(G)
        translated_sections = {}

        self.process_sections(context, text_df, section_order, "section_number", translated_sections)

        return context

    def update_translated_sections(self, translated_sections, section_number, parsed_code):
        parsed_code = [code for code in parsed_code if len(code.strip())]
        translated_sections[str(section_number)] = "\n".join(parsed_code)

    # construct a dependency graph where a node is the child of the parent, if the child has a reference on the parent.
    # One node is one paragraph (with its section).
    @staticmethod
    def construct_dependency_graph(text_df: pd.DataFrame, only_sections=False):
        G = nx.DiGraph()
        for idx, row in text_df.iterrows():
            if np.isnan(row["section_number"]):
                continue
            if row["parsed_references"] is None or str(row["parsed_references"]) == "nan":
                continue
            for parsed_reference in ast.literal_eval(row["parsed_references"]):
                if len(parsed_reference) == 0:
                    continue

                if np.isnan(row["paragraph_number"]):
                    parent = (int(row["section_number"]),)
                else:
                    parent = (int(row["section_number"]), int(row["paragraph_number"]))

                if "paragraph" not in parsed_reference or np.isnan(float(parsed_reference["paragraph"])):
                    child = (int(parsed_reference["section"]),)
                else:
                    child = (int(parsed_reference["section"]), int(parsed_reference["paragraph"]))

                if only_sections:
                    parent = parent[0]
                    child = child[0]

                G.add_edge(parent, child)

        # removes self loops
        G.remove_edges_from(nx.selfloop_edges(G))

        if only_sections:
            sections = text_df["section_number"].unique()
            for section in sections:
                if section not in G.nodes and not np.isnan(section):
                    G.add_node(int(section))

        return G

    @staticmethod
    def custom_dfs_postorder(G, source, visited):
        # Custom postorder traversal that explores nodes in sorted order
        nodes = []
        visited.add(source)

        # Sort the neighbors before visiting
        for neighbor in sorted(G[source]):
            if neighbor not in visited:
                nodes.extend(GozTextSectionHandlerDependencyTree.custom_dfs_postorder(G, neighbor, visited))
        nodes.append(source)

        return nodes

    @staticmethod
    def postorder_traversal_forest(G):
        postorder_list = []
        visited = set()

        for node in sorted(G):
            if node not in visited:
                postorder_list.extend(GozTextSectionHandlerDependencyTree.custom_dfs_postorder(G, node, visited))

        return postorder_list


class GozServiceSectionHandler(BaseSectionHandler):
    def execute(self, context: dict) -> dict:
        section_df = context["section_df"]
        text_df = context["text_df"]
        sections = section_df["section"].unique()

        translated_sections = self.get_translated_sections(text_df)

        self.process_sections(context, section_df, sections, "section", translated_sections)

        return context

    @staticmethod
    def get_translated_sections(text_df):
        translated_sections = {}
        for section_number in text_df["section_number"].unique():
            df = text_df[text_df["section_number"] == section_number]
            df = df[df["parsed_code"].apply(lambda x: x is not None and str(x) != "nan" and len(x.strip()) > 0)]
            if len(df):
                translated_sections[str(section_number)] = "\n".join(df["parsed_code"].values)
        return translated_sections


class GozServiceRulesHandler(BaseSectionHandler):
    def execute(self, context: dict) -> dict:
        section_df = context["section_df"]
        service_df = context["service_df"]

        sections = service_df["section"].unique()
        # sections = service_df[service_df["number"] <= 2410]["number"].unique()

        translated_sections = self.get_translated_sections(section_df, sections)

        self.process_sections(context, service_df, sections, "section", translated_sections)

        return context

    @staticmethod
    def get_translated_sections(section_df, sections):
        translated_sections = {}
        for section in sections:
            df = section_df[section_df["section"] == section]
            df = df[df["parsed_code"].apply(lambda x: x is not None and str(x) != "nan" and len(x.strip()) > 0)]
            if len(df):
                translated_sections[section] = "\n".join(df["parsed_code"].values)
        return translated_sections
