import argparse
import json
import os
import re

import pandas as pd
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

prompt = """Is the invoice given below in JSON format correct according to the given rule? Reason before you answer.
Answer with:
Reasoning: <your reasoning>
Answer: <yes/no>
---
Invoice: 
{invoice}
---
Rule:
>>>
{rule}
>>>
"""


def group_by_rules(df: pd.DataFrame, rule_number_column_name: str, use_commentary: bool) -> pd.DataFrame:
    df["rule_number"] = df[rule_number_column_name]
    content_columns_name = "commentary" if use_commentary and "commentary" in df.columns else "content"
    df["content"] = df[content_columns_name]
    df = df.groupby("rule_number")["content"].first().reset_index()
    return df


def combine_rules(text_df: pd.DataFrame, section_df: pd.DataFrame, service_df: pd.DataFrame, use_commentary: bool) -> pd.DataFrame:
    service_df = group_by_rules(service_df, rule_number_column_name="number", use_commentary=use_commentary)

    text_df["section_number"] = text_df["section_number"].fillna(0)
    text_df["section"] = text_df["section_number"].apply(lambda x: "§ " + str(int(x)))
    text_df = group_by_rules(text_df, rule_number_column_name="section", use_commentary=use_commentary)

    section_df = group_by_rules(section_df, rule_number_column_name="section", use_commentary=use_commentary)

    rules_df = pd.concat([text_df, section_df, service_df])
    return rules_df


def load_rule_dfs():
    text_df = pd.read_csv(os.path.join(args.code_file_path, "text_df.csv"))
    section_df = pd.read_csv(os.path.join(args.code_file_path, "section_df.csv"))
    service_df = pd.read_csv(os.path.join(args.code_file_path, "service_df.csv"))
    return text_df, section_df, service_df


def evaluate(invoices: pd.DataFrame, rules_df: pd.DataFrame, complete_service_df: pd.DataFrame) -> pd.DataFrame:

    result_df = []
    for tests_rule, invoices_group in invoices.groupby("tests_rule"):
        try:
            tests_rule = int(tests_rule)
        except:
            pass

        if tests_rule != "all":
            content = rules_df[rules_df["rule_number"] == tests_rule]["content"].values[0]

            for idx, row in invoices_group.iterrows():
                invoice = row["invoice"]
                chat_prompt = ChatPromptTemplate.from_messages([HumanMessagePromptTemplate.from_template(prompt)])
                chain = LLMChain(
                    llm=ChatOpenAI(temperature=0, model_name="gpt-4"),
                    prompt=chat_prompt,
                    verbose=True,
                )

                response = chain.predict(invoice=str(invoice), rule=content)
                print(response)

                # parse the answer
                answer = re.search(r"Answer: (.*)", response).group(1)
                prediction = True if answer.lower() == "yes" else False

                result_df.append({"tests_rule": tests_rule, "invoice": invoice, "result": answer, "is_correct": row["is_correct"], "explanation": row["explanation"], "prediction": prediction, "llm_response": response, "llm_input": chat_prompt.format(invoice=invoice, rule=content)})

    return pd.DataFrame(result_df)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--OPENAI_API_KEY', type=str, required=True)
    parser.add_argument('--invoices_json_file', type=str, default='./resources/evaluation_goz/invoices/invoices.json')
    parser.add_argument('--code_file_path', type=str, default='./resources/parsed_text/parsed_rules_with_commentary')
    parser.add_argument('--output_file_path', type=str, default='./resources/evaluation_goz/results')
    parser.add_argument('--use_commentary', action="store_true")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ["OPENAI_API_KEY"] = args.OPENAI_API_KEY
    if not os.path.exists(args.invoices_json_file):
        raise ValueError(f"There are no invoice JSON files in {args.invoices_json_file}")

    with open(args.invoices_json_file, "r", encoding="utf-8") as f:
        invoices = json.load(f)

    invoices = pd.DataFrame(invoices)
    text_df, section_df, service_df = load_rule_dfs()
    rules_df = combine_rules(text_df, section_df, service_df, args.use_commentary)

    if not os.path.exists(args.output_file_path):
        os.makedirs(args.output_file_path)

    for i in range(1, 5):
        result_df = evaluate(invoices, rules_df, service_df)
        result_df.to_csv(os.path.join(args.output_file_path, f"result_baseline_with_commentary_{i}.csv"))

        wrong_df = result_df[result_df["prediction"] != result_df["is_correct"]]
        rules = result_df["tests_rule"].unique()
        wrong_rules = wrong_df["tests_rule"].unique()
        accuracy = (len(rules) - len(wrong_rules)) / len(rules)
        print("accuracy", accuracy)