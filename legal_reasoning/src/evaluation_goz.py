import sys

sys.path.append('../../')

import argparse
import json
import os
import re

import pandas as pd
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

import prolog_utils
from code_execution import PrologExecutor
from invoice_json_to_prolog import json_to_prolog


def group_by_rules(df: pd.DataFrame, rule_number_column_name: str) -> pd.DataFrame:
    def agg_code(x):
        x = [v for v in x.values if v]
        return "\n".join(x)

    df["parsed_code"] = df["parsed_code"].fillna("")
    df["rule_number"] = df[rule_number_column_name]
    df = df.groupby("rule_number").agg({"parsed_code": agg_code}).reset_index()
    return df


def combine_rules(text_df: pd.DataFrame, section_df: pd.DataFrame, service_df: pd.DataFrame) -> pd.DataFrame:
    service_df = group_by_rules(service_df, rule_number_column_name="number")

    text_df["section_number"] = text_df["section_number"].fillna(0)
    text_df["section"] = text_df["section_number"].apply(lambda x: "§ " + str(int(x)))
    text_df = group_by_rules(text_df, rule_number_column_name="section")

    section_df = group_by_rules(section_df, rule_number_column_name="section")

    rules_df = pd.concat([text_df, section_df, service_df])
    return rules_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--OPENAI_API_KEY', type=str, required=True)
    parser.add_argument('--invoices_json_file', type=str, default='./resources/evaluation_goz/invoices/invoices.json')
    parser.add_argument('--code_file_path', type=str, default='./resources/results/gpt-4_v0')
    parser.add_argument('--output_file_path', type=str, default='./resources/evaluation_goz/results')
    args = parser.parse_args()
    return args


def load_rule_dfs():
    text_df = pd.read_csv(os.path.join(args.code_file_path, "text_df.csv"), index_col=0)
    section_df = pd.read_csv(os.path.join(args.code_file_path, "section_df.csv"), index_col=0)
    service_df = pd.read_csv(os.path.join(args.code_file_path, "service_df.csv"), index_col=0)
    return text_df, section_df, service_df


def evaluate(invoices: pd.DataFrame, rules_df: pd.DataFrame, complete_service_df: pd.DataFrame) -> pd.DataFrame:
    code_executor = PrologExecutor()

    result_df = []
    for tests_rule, invoices_group in invoices.groupby("tests_rule"):
        try:
            tests_rule = int(tests_rule)
        except:
            pass

        if tests_rule != "all":
            code = rules_df[rules_df["rule_number"] == tests_rule]["parsed_code"].values[0]

            for idx, row in invoices_group.iterrows():
                prolog_invoice = json_to_prolog(row["invoice"], complete_service_df)
                # remove ending .
                prolog_invoice = prolog_invoice[:-1]
                prolog_invoice = f"example_invoice({prolog_invoice})."
                code_with_invoice = code + "\n" + prolog_invoice
                query = "example_invoice(X), is_invoice_invalid(X)."
                interpreter_result = code_executor.execute(code=code_with_invoice, query=query)
                result = prolog_utils.parse_prolog_output(interpreter_result, query)

                count = 0
                while result == "Error" and len(code.strip()):
                    output = re.sub(r"Warning:.*\n", "", interpreter_result)

                    correction_prompt = (
                        f"The prolog code \n```prolog\n{code}\n```\nhas produced some Errors.\n"
                        f"Please fix the Errors by changing the code without altering its meaning.\n Errors:\n{output}\n"
                        f"Write \ncode:\n ```prolog\n<CODE>\n```\nfor the correction of the code.")
                    chat_prompt = ChatPromptTemplate.from_messages(
                        [HumanMessagePromptTemplate.from_template(correction_prompt)])
                    chain = LLMChain(
                        llm=ChatOpenAI(temperature=0, model_name="gpt-4"),
                        prompt=chat_prompt,
                        verbose=True,
                    )
                    response = chain.predict()
                    print(response)
                    code = response.split("```prolog")[1].split("```")[0].strip()
                    code_with_invoice = code + "\n" + prolog_invoice
                    interpreter_result = code_executor.execute(code=code_with_invoice, query=query)
                    result = prolog_utils.parse_prolog_output(interpreter_result, query)
                    count += 1
                    if count > 5:
                        print("!!!!!!!!Correction module chain restarted more than 5 times.!!!!!!!")
                        break

                prediction = str(result) == "False"

                result_df.append({"tests_rule": tests_rule, "invoice": row["invoice"], "result": result,
                                  "interpreter_result": interpreter_result, "code": code_with_invoice,
                                  "is_correct": row["is_correct"], "explanation": row["explanation"],
                                  "prediction": prediction})

    return pd.DataFrame(result_df)


if __name__ == '__main__':
    args = parse_args()
    os.environ["OPENAI_API_KEY"] = args.OPENAI_API_KEY
    if not os.path.exists(args.invoices_json_file):
        raise ValueError(f"There are no invoice JSON files in {args.invoices_json_file}")

    with open(args.invoices_json_file, "r", encoding="utf-8") as f:
        invoices = json.load(f)

    invoices = pd.DataFrame(invoices)

    text_df, section_df, service_df = load_rule_dfs()
    rules_df = combine_rules(text_df, section_df, service_df)

    if not os.path.exists(args.output_file_path):
        os.makedirs(args.output_file_path)

    result_df = evaluate(invoices, rules_df, service_df)
    result_df.to_csv(os.path.join(args.output_file_path, f"result_{args.code_file_path.split('/')[-1]}.csv"))

    wrong_df = result_df[result_df["prediction"] != result_df["is_correct"]]
    rules = result_df["tests_rule"].unique()
    wrong_rules = wrong_df["tests_rule"].unique()
    accuracy = (len(rules) - len(wrong_rules)) / len(rules)
    print("accuracy", accuracy)
