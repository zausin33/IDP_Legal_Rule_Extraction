import argparse
import json
import os
from typing import Dict, Any

import langchain
import numpy as np
import pandas as pd
from langchain.chains import FlareChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from tqdm import tqdm

from arctic_sea import graph_utils
from arctic_sea.graph_utils import create_graph_from_llm_output, evaluate
from src.langchain_modules import knowledge_base
from src.langchain_modules import prompts
from src.langchain_modules.output_parser import AnswerOutputParser


def main(args: Dict[str, Any]):
    print(args)
    build_vector_store = args.get("build_vector_store", False)
    vector_store_size = args.get("vector_store_size", "small")

    use_knowledge_base = args.get("use_knowledge_base", False)
    use_multi_query_retriever = args.get("use_multi_query_retriever", False)
    use_flare_chain = args.get("use_flare_chain", False)
    use_contextual_compression = args.get("use_contextual_compression_retriever", False)

    shot_setting = args.get("shot_setting", "zero_shot")
    model_name = args.get("model_name", "gpt-3.5-turbo")

    # check args
    if use_flare_chain:
        use_knowledge_base = True
        if use_multi_query_retriever:
            raise ValueError("use_multi_query_retriever and use_flare_chain cannot be used together.")
        if use_contextual_compression:
            raise ValueError("use_contextual_compression and use_flare_chain cannot be used together.")

    if use_contextual_compression or use_multi_query_retriever:
        use_knowledge_base = True

    # load some variables
    with open("./data/graph_variable_name_mapping.json") as f:
        variable_name_mapping = json.load(f)
    name_variable_mapping = {v: k for k, v in variable_name_mapping.items()}
    variables = list(name_variable_mapping.keys())
    variable_combinations = [(variable1, variable2) for variable1 in variables for variable2 in variables if
                             variable1 != variable2]
    feature_names = list(variable_name_mapping.keys())
    gt_adj = np.loadtxt("./data/ground_truth_graph")
    gt_G = graph_utils.create_graph(feature_names, gt_adj, do_draw=False)

    # build vector store
    if use_knowledge_base:
        if not build_vector_store and not os.path.exists(f"./vector_stores/{vector_store_size}"):
            print(f"Vector store {vector_store_size} does not exist -> building it")
            build_vector_store = True

        if build_vector_store:
            vector_db = knowledge_base.build_vector_store_from_folder(
                folder_path=f"./knowledge_base/{vector_store_size}",
                persist_directory=f"./vector_stores/{vector_store_size}", tokens_per_chunk=3000, chunk_overlap=100)
        else:
            vector_db = knowledge_base.load_vector_store(persist_directory=f"./vector_stores/{vector_store_size}")



    # define prompt


    # build chain
    llm = ChatOpenAI(temperature=0, model_name=model_name)

    if use_knowledge_base:
        retriever = vector_db.as_retriever(search_kwargs={"k": 8})
        if use_multi_query_retriever:
            retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
        if use_contextual_compression:
            compressor = LLMChainExtractor.from_llm(llm)
            retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

    if use_flare_chain:
        chain = FlareChain.from_llm(
            llm,
            retriever=vector_db.as_retriever(k=2),
            max_generation_len=164,
            min_prob=0.3,
            )
    else:
        chat_prompt = ChatPromptTemplate.from_messages([
            prompts.SYSTEM_PROMPT_ARCTIC_SEA,
            prompts.YES_NO_PROMPT_WITH_CONTEXT_NEW if use_knowledge_base else prompts.YES_NO_PROMPT_WITHOUT_CONTEXT_NEW,
        ])
        chain = LLMChain(llm=llm, prompt=chat_prompt)

    answers = []

    # run chain
    for X, Y in tqdm(variable_combinations, total=len(variable_combinations)):
        if use_knowledge_base:
            if use_flare_chain:
                llm_input = prompts.FLARE_PROMPT.format(X=X, Y=Y).content
                ai_text = chain.run(llm_input)
            else:
                query = f"Can you tell me about the relationship between {X} and {Y}?"
                documents = retriever.get_relevant_documents(query)
                context = "\n\n".join(doc.page_content for doc in documents)
                ai_text = chain.run(X=X, Y=Y, context=context)
                llm_input = chain.prompt.format(X=X, Y=Y, context=context)
        else:
            ai_text = chain.run(X=X, Y=Y)
            llm_input = chain.prompt.format(X=X, Y=Y)

        answers.append(ai_text)

    output_parser = AnswerOutputParser.from_llm(llm, answer_options=["yes", "no"], with_retry_chain=False)
    parsed_results = [output_parser.parse(text, "") for idx, text in enumerate(answers)]
    input_list = [{"X": x, "Y": y} for x, y in variable_combinations]

    G = create_graph_from_llm_output(parsed_results, input_list, name_variable_mapping, edges_yes_no=True, bidirectional=True, gt_G=gt_G)
    evaluate(gt_G, G)

    result_df = pd.DataFrame({"answer": answers, "parsed_results": parsed_results, "input": input_list, "llm_input": llm_input})
    result_name = f"result_{shot_setting}_{model_name}"
    if use_knowledge_base:
        result_name += f"_kb_{vector_store_size}"
    if use_multi_query_retriever:
        result_name += "_multi_query"
    if use_flare_chain:
        result_name += "_flare_chain"
    if use_contextual_compression:
        result_name += "_contextual_compression"
    result_df.to_csv(f"./data/results/{result_name}.csv")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build_vector_store", action="store_true")
    parser.add_argument("--vector_store_size", type=str, default="small", choices=["small", "large"])
    parser.add_argument("--use_knowledge_base", action="store_true")
    parser.add_argument("--use_multi_query_retriever", action="store_true")
    parser.add_argument("--use_flare_chain", action="store_true")
    parser.add_argument("--use_contextual_compression_retriever", action="store_true")
    parser.add_argument("--shot_setting", type=str, default="zero_shot", choices=["zero_shot", "one_shot", "few_shot"])
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

