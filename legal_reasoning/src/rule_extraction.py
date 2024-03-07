import argparse
import ast
import os
from typing import List

import pandas as pd
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

prompt = lambda language: """Given following text which is a part of the German GOZ:
```
{{ text }}
```
{{ references }}

Extract from this text only rules that both state conditions or restrictions that should be adhered to when creating an invoice and can be checked automatically by some code, corresponding to an invoice with the following schema:
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "Services": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "ServiceNumber": {
            "type": "integer"
          },
          "DateOfTreatment": {
            "type": "string",
            "format": "date"
          },
          "Multiplier": {
            "type": "number"
          },
          "PointScore": {
            "type": "integer"
          },
          "Charge": {
            "type": "number"
          },
          "Description": {
            "type": "string"
          },
          "Justification": {
            "type": "string"
          },
          "Tooth": {
            "type": "integer"",
          }
        },
        "required": ["ServiceNumber", "DateOfTreatment", "Multiplier", "PointScore", "Charge", "Description"]
      }
    },        
    "MaterialCosts": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "GOZ_Section": {
            "type": "string"
          },
          "DateOfTreatment": {
            "type": "string",
            "format": "date"
          },
          "Multiplier": {
            "type": "number"
          },
          "Count": {
            "type": "integer"
          },
          "Charge": {
            "type": "number"
          },
          "Description": {
            "type": "string"
          },
          "Justification": {
            "type": "string"
          },
          "Teeth": {
            "type": "array",
            "items": {
              "type": "integer"
            }
          }
        },
        "required": ["DateOfTreatment", "Multiplier", "Count", "Charge", "Description"]
      }
    },            
    "InvoiceDate": {
      "type": "string",
      "format": "date"
    },
    "InvoiceAmount": {
      "type": "number"
    }
  },
  "required": ["services", "InvoiceDate", "InvoiceAmount"]
}
Extract only rules from the text that can be applied to a invoice with this structure.
If you do not find such rules, just return <NO RULES>.
Return the rule in text form and in the language LANGUAGE.
Separate the rules with ---.
e.g.:
Rules:
---
Rule 1
---
Rule 2
---
...


Rules:
""".replace("LANGUAGE", language)


def preprocess_text_into_rule(text: str, references: str, language: str, open_ai_model_name:str) -> List[str]:
    if not len(text.strip()):
        return []

    chat_prompt = ChatPromptTemplate.from_messages([HumanMessagePromptTemplate.from_template(prompt(language), template_format="jinja2")])
    translation_chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model_name=open_ai_model_name),
        prompt=chat_prompt,
        verbose=True,
    )

    if references:
        references = (
            "\nOnly for making the rule extraction easier, here are the referenced sections:\n"
            "```"
            f"{references}"
            "```"
        )
    else:
        references = ""

    response = translation_chain.predict(text=text, references=references)
    print(response)

    if "<NO RULES>" in response:
        return []

    rules = []
    for part in response.split("---")[1:]:
        part = part.strip()
        if len(part):
            rules.append(part)

    return rules


def get_references(parsed_references: str, text_df: pd.DataFrame, current_section: str, current_paragraph: str) -> str:
    # get referenced section and paragraph
    # check that it is not a reference on himself
    # return the referenced section and paragraph

    parsed_references = ast.literal_eval(parsed_references) if parsed_references and str(parsed_references) != "nan" else []

    references = ""
    for reference in parsed_references:
        if "section" not in reference or not reference["section"]:
            continue
        section = reference["section"]
        paragraph = None
        if "paragraph" in reference:
            paragraph = reference["paragraph"]

        if section == current_section:
            if not current_paragraph or current_paragraph == paragraph:
                continue

        if paragraph:
            ref = text_df[(text_df["section_number"] == float(section)) & (text_df["paragraph_number"] == float(paragraph))]["content"]
            if len(ref) == 0:
                continue
            ref = f"§ {section} ({paragraph})\n{ref.values[0]}"
        else:
            ref = text_df[(text_df["section_number"] == float(section))]["content"]
            if len(ref) == 0:
                continue
            ref = '\n'.join(ref.values)
            ref = f"§ {section}\n{ref}"
        references += "\n" + ref + "\n"

    return references


def process_df(df: pd.DataFrame, text_df: pd.DataFrame, language: str, open_ai_model_name) -> pd.DataFrame:
    new_df = []
    for i, row in df.iterrows():
        text = row["content"]
        if not text or str(text) == "nan":
            continue

        section_number = row["section_number"] if "section_number" in row else row["section"]
        paragraph_number = row["paragraph_number"] if "paragraph_number" in row else None
        references = get_references(row["parsed_references"], text_df, section_number, paragraph_number)

        with get_openai_callback() as cb:
            rules = preprocess_text_into_rule(text, references, language, open_ai_model_name)
            for rule in rules:
                row_dict = row.to_dict()
                row_dict["rule"] = rule
                row_dict["extr_rules_calls"] = cb.successful_requests
                row_dict["extr_rules_prompt_tokens"] = cb.prompt_tokens
                row_dict["extr_rules_completion_tokens"] = cb.completion_tokens
                row_dict["extr_rules_total_cost"] = cb.total_cost
                new_df.append(row_dict)

            if len(rules) == 0:
                row_dict = row.to_dict()
                row_dict["rule"] = None
                row_dict["extr_rules_calls"] = cb.successful_requests
                row_dict["extr_rules_prompt_tokens"] = cb.prompt_tokens
                row_dict["extr_rules_completion_tokens"] = cb.completion_tokens
                row_dict["extr_rules_total_cost"] = cb.total_cost
                new_df.append(row_dict)

    return pd.DataFrame(new_df)



def preprocess_text_segments(text_df, service_df, section_df, language, open_ai_model_name):
    text_df = process_df(text_df, text_df, language, open_ai_model_name)
    service_df = process_df(service_df, text_df, language, open_ai_model_name)
    section_df = process_df(section_df, text_df, language, open_ai_model_name)
    return text_df, service_df, section_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--OPENAI_API_KEY', type=str, required=True)
    parser.add_argument('--file_path', type=str, default='./resources/parsed_text/de')
    parser.add_argument('--output_file_path', type=str, default='./resources/parsed_text/parsed_rules_v2')
    parser.add_argument('--open_ai_model_name', type=str, default='gpt-4')
    parser.add_argument('--language', type=str, default="german")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ["OPENAI_API_KEY"] = args.OPENAI_API_KEY
    service_df = pd.read_csv(os.path.join(args.file_path, "service_df.csv"))
    text_df = pd.read_csv(os.path.join(args.file_path, "text_df.csv"))
    section_df = pd.read_csv(os.path.join(args.file_path, "section_df.csv"))

    text_df, service_df, section_df = preprocess_text_segments(text_df, service_df, section_df, args.language, args.open_ai_model_name)

    if not os.path.exists(args.output_file_path):
        os.makedirs(args.output_file_path)

    text_df.to_csv(f"{args.output_file_path}/text_df.csv", index=False)
    service_df.to_csv(f"{args.output_file_path}/service_df.csv", index=False)
    section_df.to_csv(f"{args.output_file_path}/section_df.csv", index=False)