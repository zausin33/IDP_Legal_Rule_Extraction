# Important: set swipl command via env. file

from __future__ import annotations

import sys
sys.path.append('../../')

import argparse
import os
import random

import pandas as pd

from code_execution import PrologExecutor
from model.llm import LLMPrompter
from utils import Config, ChatModels

random.seed(42)


def load_preprocessed_goz(path: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    service_df = pd.read_csv(os.path.join(path, "service_df.csv"))
    text_df = pd.read_csv(os.path.join(path, "text_df.csv"))
    section_df = pd.read_csv(os.path.join(path, "section_df.csv"))

    return text_df, service_df, section_df


# Main Application Flow

def main(config: Config, input_file_path: str, output_file_path: str):
    code_executor = PrologExecutor()
    llm_prompter = LLMPrompter(
        config=config,
        code_executor=code_executor
    )

    text_df, service_df, section_df = load_preprocessed_goz(input_file_path)
    text_df, service_df, section_df = llm_prompter.prompt_llm_for_translation({"text_df": text_df, "service_df": service_df, "section_df": section_df})

    goz_commentary = "-goz_commentary" if config.use_goz_commentary else ""
    path = os.path.join(output_file_path, f"{config.open_ai_model_name}{goz_commentary}_v0")

    # check if path exists, if it exists, create new folder with increasing numbers
    if os.path.exists(path):
        i = 1
        while os.path.exists(path):
            path = os.path.join(output_file_path, f"{config.open_ai_model_name}{goz_commentary}_v{i}")
            i += 1

    os.makedirs(path)

    text_df.to_csv(os.path.join(path, "text_df.csv"))
    service_df.to_csv(os.path.join(path, "service_df.csv"))
    section_df.to_csv(os.path.join(path, "section_df.csv"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--OPENAI_API_KEY', type=str, required=True)
    parser.add_argument('--file_path', type=str, default='./resources/parsed_text/parsed_rules_with_commentary')
    parser.add_argument('--output_file_path', type=str, default='./resources/results/')
    parser.add_argument('--use_commentary', action="store_true")
    parser.add_argument('--open_ai_model_name', type=str, default=ChatModels.GPT_4.value, choices=[model.value for model in ChatModels])
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ["OPENAI_API_KEY"] = args.OPENAI_API_KEY
    config = Config(
        open_ai_model_name=args.open_ai_model_name,
        use_examples_for_translation=False,
        with_human_editing=False,
        extract_rules_first=True,
        use_goz_commentary=args.use_commentary,
    )

    main(config, args.file_path, args.output_file_path)
