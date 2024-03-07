import networkx
import os
import pickle
import tqdm
from langchain import LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import LLMResult
from langchain.vectorstores import VectorStore
from typing import Callable

from arctic_sea.graph_utils import create_graph_from_llm_output, evaluate
from src.langchain_modules.output_parser import AnswerOutputParser

RUN_PATH = "./runs"


def create_causal_reasoning_chain(
        llm: BaseLanguageModel,
        run_name: str,
        system_prompt: SystemMessagePromptTemplate,
        human_prompt: HumanMessagePromptTemplate,
        name_variable_mapping: dict[str, str],
        yes_no_prompt: bool = False,
        with_context: bool = False,
        get_context: Callable[[str], str] = None,
        overwrite_existing: bool = False,
) -> tuple[LLMChain, list[dict[str, str]]]:
    if with_context and get_context is None:
        raise ValueError("If with_context is True, then get_context must be provided.")

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    chain = LLMChain(llm=llm, prompt=chat_prompt, verbose=True)
    input_list = __create_input_values(run_name, overwrite_existing, get_context, name_variable_mapping, with_context, yes_no_prompt)
    return chain, input_list


def __create_input_values(run_name, overwrite_existing, get_context, name_variable_mapping, with_context, yes_no_prompt) -> list[dict[str, str]]:
    if not overwrite_existing and os.path.exists(os.path.join(RUN_PATH, run_name)):
        print(f"Run {run_name} already exists. Skipping generating input.")
        return load_data(run_name)[0]

    variables = list(name_variable_mapping.keys())
    if yes_no_prompt:
        variable_combinations = [(variable1, variable2) for variable1 in variables for variable2 in variables if
                                 variable1 != variable2]
    else:
        variable_combinations = [(variables[i], variables[j]) for i in range(len(variables)) for j in
                                 range(i + 1, len(variables))]

    input_list = [{"X": x, "Y": y} for x, y in variable_combinations]

    if with_context:
        for idx, input in tqdm.tqdm(enumerate(input_list), desc=f"Context retrival", total=len(input_list)):
            search_query = f"{input['X']} {input['Y']}"
            context = get_context(text=search_query)
            input_list[idx]["context"] = context
    return input_list


def get_context_from_vector_store(text: str, vector_store: VectorStore) -> str:
    context = vector_store.similarity_search(text, k=6)
    context = "\n\n".join([c.page_content for c in context])
    return context


def run_chain(chain: LLMChain, input_list: list[dict[str, str]], run_name: str, overwrite_existing: bool = False) -> LLMResult:
    if not overwrite_existing and os.path.exists(os.path.join(RUN_PATH, run_name)):
        print(f"Run {run_name} already exists. Skipping generating output.")
        return load_data(run_name)[1]

    output_list = chain.generate(input_list)
    __save_data(input_list, output_list, run_name)
    return output_list


def __save_data(input_list: list[dict[str, str]], output_list:LLMResult, run_name):
    if not os.path.exists(RUN_PATH):
        os.mkdir(RUN_PATH)

    run_path = os.path.join(RUN_PATH, run_name)
    if not os.path.exists(run_path):
        os.mkdir(run_path)

    output_path = os.path.join(run_path, "output.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(output_list, f)

    input_path = os.path.join(run_path, "input.pkl")
    with open(input_path, "wb") as f:
        pickle.dump(input_list, f)

    print(f"Saved run to {run_path}")


def load_data(run_name: str) -> tuple[list[dict[str, str]], LLMResult]:
    run_path = os.path.join(RUN_PATH, run_name)
    output_path = os.path.join(run_path, "output.pkl")
    input_path = os.path.join(run_path, "input.pkl")
    with open(output_path, "rb") as f:
        output_list = pickle.load(f)
    with open(input_path, "rb") as f:
        input_list = pickle.load(f)
    return input_list, output_list


def create_run_evaluate_chain(
        run_name: str,
        name_variable_mapping: dict[str, str],
        gt_G: networkx.Graph,
        yes_no_prompt: bool,
        with_context: bool,
        system_prompt: SystemMessagePromptTemplate,
        human_prompt: HumanMessagePromptTemplate,
        get_context: Callable[[str], str] = None,
        bidirectional_edges: bool = True,
        overwrite_existing: bool = False,
        with_retry_chain: bool = False,
):
    chat = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain, input_list = create_causal_reasoning_chain(
        llm=chat,
        run_name=run_name,
        system_prompt=system_prompt,
        human_prompt=human_prompt,
        name_variable_mapping=name_variable_mapping,
        yes_no_prompt=yes_no_prompt,
        with_context=with_context,
        get_context=get_context,
        overwrite_existing=overwrite_existing,
    )

    if yes_no_prompt:
        answer_options = ["yes", "no"]
    elif bidirectional_edges:
        answer_options = ["A", "B", "C", "D"]
    else:
        answer_options = ["A", "B", "C"]

    output_parser = AnswerOutputParser.from_llm(chat, answer_options=answer_options, with_retry_chain=with_retry_chain)
    print("Chain Check: \n", chain.prompt.format_prompt(**input_list[0]).to_string(), "\n\n")
    result = run_chain(chain, input_list, run_name=run_name)
    parsed_results = [output_parser.parse(gen[0].text, chain.prompt.format_prompt(**input_list[idx]).to_string()) for idx, gen in enumerate(result.generations)]
    print(len(parsed_results))
    G = create_graph_from_llm_output(parsed_results, input_list, name_variable_mapping, edges_yes_no=yes_no_prompt, bidirectional=bidirectional_edges, gt_G=gt_G)
    evaluate(gt_G, G)

    return G, parsed_results, result, input_list
