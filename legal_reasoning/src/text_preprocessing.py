import argparse
import os
import re
from typing import List, Dict

import pandas as pd
from PyPDF2 import PdfReader
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
import pdfplumber

"""
Splits the GOZ pdf into sections and extracts the table with the services.
Also extracts references to other sections and subsections.
Translation of the text is also possible.
"""

rule_extraction_prompt = """Given following text which is a part of the German GOZ:
```
{text}
```

Extract references to other paragraphs if mentioned in the text.

Write the result as a list of references like for example:
References: §5 (1), §4 (2)"""


def goz_to_sections(file_path: str, language: str = "de"):
    if language != "de" and language != "en":
        raise NotImplementedError("Only German and English language is supported. Choose either 'de' or 'en'")

    text_list = []
    reader = PdfReader(file_path)
    for page in reader.pages:
        text = page.extract_text()
        text_list.append(text)

    # remove first two cover pages
    text_list = text_list[2:]
    # concatenate the text
    text_list = "\n".join(text_list)
    # split always when a new paragraph starts with new line and § and digit. Keeps the paragraph number in the next split.
    text_list = re.split(r"(?=\n§\s?\d+)", text_list)
    # remove the GOZ footer
    text_list = [re.sub(goz_dictionary["footer_regex"][language], "", text) for text in text_list]

    # extract the table
    last_paragraph, table = text_list[-1].split(goz_dictionary["table_beginning"][language])
    text_list[-1] = last_paragraph

    # remove a \n, but only when it is not followed by another \n and in the next line is not (digit) and when it is not preceeded by a "-"
    text_list = [re.sub(r"(?<!-)\s?\n\s?(?!\n)(?!\(\d)", " ", text) for text in text_list]

    # remove a -\n and concatenate the word
    text_list = [re.sub(r"\s?-\n", "", text) for text in text_list]

    # parse text list into dataframe
    text_df = []

    for section in text_list:
        section = section.strip()
        # Extracting the section header
        section_header = re.search(r"(^§\s?\d+ [^\n]+)", section)
        if not section_header:
            text_df.append({"section": pd.NA, "subsection": pd.NA, "content": section})
            continue

        section_header = section_header.group(1)
        new_section = section.replace(section_header, "").strip()

        if not new_section:
            section_header = re.search(r"(^§\s?\d+\s+\S+\s)", section)
            section_header = section_header.group(1)
            section = section.replace(section_header, "").strip()
        else:
            section = new_section

        # Splitting the main section based on subsections
        subsections = re.split(r"(?=\(\d+\))", section)

        for sub in subsections:
            subsection_header = re.search(r"\(\d+\)", sub)
            if subsection_header:
                subsection_content = sub.replace(subsection_header.group(0), "").strip()
                text_df.append({"section": section_header, "subsection": subsection_header.group(0),
                                "content": subsection_content})
            elif sub.strip():
                text_df.append({"section": section_header, "subsection": pd.NA, "content": sub})

    text_df = pd.DataFrame(text_df)

    sections = extract_sections(table)
    parsed_data_list = []
    section_list = []

    for section in sections:
        parsed_data, section_character, info_text = extract_table_rows(section)
        parsed_data_list.extend(parsed_data)
        section_list.append({"section": section_character, "content": info_text})

    section_df = pd.DataFrame(section_list)
    service_df = pd.DataFrame(parsed_data_list)

    # Splitting the 'values' column into multiple columns
    cols = ['Punktzahl', '1.0-fach', '2.3-fach', '3.5-fach']
    service_df[cols] = pd.DataFrame(service_df['values'].tolist(), index=service_df.index)

    # Convert columns to string, replace comma with dot and convert to float
    for col in cols:
        service_df[col] = service_df[col].str.replace(',', '.').astype(float)

    # Drop the original 'values' column
    service_df.drop(columns=['values'], inplace=True)

    return text_df, service_df, section_df


goz_dictionary = {
    "footer_regex": {"en": r"\(GOZ\)\s?(\d\d\d\d)?",
                     "de": r"Gebührenordnung für Zahnärzte \(GOZ\)\s?(\d\d\d\d)?\s?\d?\d?"},
    "table_beginning": {"en": "", "de": "Gebührenverzeichnis für zahnärztliche Leistungen >>"},
    "table_header": {"en": "", "de": "Nummer Leistung Punktzahl 1,0-fach"},
}


def extract_sections(table):
    # Split the raw text into lines
    lines = table.strip().split("\n")

    # Regex pattern to identify sections
    section_pattern = re.compile(r'^[A-Z]\s+\w')
    allgemeine_bestimmungen_pattern = "Allgemeine Bestimmungen"
    table_header = "Nummer Leistung Punktzahl"

    sections = []
    section_info = []
    info_for_next_section = []
    capture_info_for_next_section = False
    current_section = None
    in_table = False

    # Iterate over lines to identify and capture section names
    for line in lines:
        line = line.strip()

        if section_pattern.match(line):
            # If we found a section and there was a previous section, append to sections list
            if current_section:
                sections.append({
                    "section": current_section,
                    "info": "\n".join(section_info)
                })
                section_info = info_for_next_section
                info_for_next_section = []
            current_section = line
            in_table = False
            capture_info_for_next_section = False
            continue

        if table_header in line and not capture_info_for_next_section:
            # Mark that we're in a table and add table header
            in_table = True
            section_info.append(line)
            continue

        if allgemeine_bestimmungen_pattern in line and in_table:
            # Start capturing the info for the section header
            info_for_next_section.append(line)
            capture_info_for_next_section = True
            continue

        if capture_info_for_next_section:
            info_for_next_section.append(line)
            continue

        if current_section:
            section_info.append(line)

    # Append the last section if exists
    if current_section:
        sections.append({
            "section": current_section,
            "info": "\n".join(section_info)
        })

    return sections


def parse_text(text, section_character):
    # Splitting blocks using regular expressions
    blocks = re.split(r'(^|\n)(?=\d{4}\s)', text)
    blocks = [block.strip() for block in blocks if block.strip()]
    append_on_next_block = ""

    parsed_data = []

    for idx, block in enumerate(blocks):
        if append_on_next_block:
            block = append_on_next_block + " " + block
            append_on_next_block = ""
        # Extracting number
        number_match = re.search(r'^\d{4}', block)
        if not number_match:
            raise Exception("Number not found in block: " + block)
        number = number_match.group()

        if idx + 1 < len(blocks):
            next_number_match = re.search(r'^\d{4}', blocks[idx + 1])
            if not next_number_match:
                append_on_next_block = block
                continue
            next_number = next_number_match.group()
            if float(next_number) <= float(number) or float(next_number) > float(number) + 100:
                append_on_next_block = block
                continue

        if "Teilleistungen nach den Nummern" in block:
            block = block.split("Teilleistungen nach den Nummern")[0]

        # Extracting the four values
        number_value_pattern = re.compile(r'(\d+[,\d+]*)')
        values = re.findall(number_value_pattern, block)

        # Extracting description
        start_idx = number_match.end()

        n_values = 4 if not section_character.startswith("L") else 2

        if len(values) < n_values:
            end_idx = len(block)
        else:
            end_idx = block.rfind(values[-n_values])

        if len(number_value_pattern.findall(block[end_idx:])) != n_values and number in ["0120", "2230", "2240", "5050",
                                                                                         "5060", "5240"]:
            values = []
            description = block[start_idx:].strip()
        else:
            # if the last 3 values are no numbers with two digits after the comma, we continue
            if not section_character.startswith("L") and (
                    not re.match(r'\d+,\d{2}', values[-3]) or not re.match(r'\d+,\d{2}', values[-2]) or not re.match(
                    r'\d+,\d{2}', values[-1])):
                append_on_next_block = block
                continue
            if section_character.startswith("L") and not re.match(r'\d+,\d{2}', values[-1]):
                append_on_next_block = block
                continue
            description = block[start_idx:end_idx].strip()

        parsed_data.append({
            'number': number,
            'content': description,
            'values': values[-n_values:],
            'section': section_character,
        })

    return parsed_data


# Extract table rows from the section info
def extract_table_rows(section):
    info = section["info"]
    section_character = section["section"]
    info_rows = []
    current_rows = []
    parsed_data_list = []
    capture = False

    lines = info.split("\n")
    count = 0

    while count < len(lines):
        line = lines[count]
        if "Nummer Leistung Punktzahl" in line:
            capture = True
            if len(current_rows):
                table_text = "\n".join(current_rows)
                parsed_data = parse_text(table_text, section_character)
                parsed_data_list.extend(parsed_data)
                current_rows = []
            # skip table header
            count += 4
            continue

        if section_character.startswith("L") and line.startswith("Anlage 2"):
            capture = False
            break

        if capture:
            current_rows.append(line)
        else:
            info_rows.append(line)
        count += 1

    info_text = "\n".join(info_rows)
    table_text = "\n".join(current_rows)
    parsed_data = parse_text(table_text, section_character)
    parsed_data_list.extend(parsed_data)
    return parsed_data_list, section_character, info_text


def preprocess_text_segments(text_df, service_df, section_df, open_ai_model_name: str):
    cols = ["reference", "extr_ref_calls", "extr_ref_prompt_tokens", "extr_ref_completion_tokens", "extr_ref_total_cost"]
    text_df[cols] = text_df["content"].apply(lambda content: extract_references(content, open_ai_model_name)).apply(pd.Series)
    text_df["section_number"] = text_df["section"].apply(extract_section_number)
    text_df["paragraph_number"] = text_df["subsection"].apply(extract_paragraph_number)
    text_df["parsed_references"] = text_df["reference"].apply(parse_references)
    text_df["parsed_references"] = text_df.apply(
        lambda row: complete_reference(row["parsed_references"], row["section_number"], row["paragraph_number"]),
        axis=1)

    service_df[cols] = service_df["content"].apply(lambda content: extract_references(content, open_ai_model_name)).apply(pd.Series)
    service_df["parsed_references"] = service_df["reference"].apply(
        lambda row: parse_references(row, must_contain_section=True))
    service_df["section"] = service_df["section"].apply(lambda section: re.sub(r"\s+", r"_", section))

    section_df[cols] = section_df["content"].apply(lambda content: extract_references(content, open_ai_model_name)).apply(pd.Series)
    section_df["parsed_references"] = section_df["reference"].apply(
        lambda row: parse_references(row, must_contain_section=True))
    section_df["section"] = section_df["section"].apply(lambda section: re.sub(r"\s+", r"_", section))
    section_df["section_letter"] = section_df["section"].apply(lambda section: section.split("_")[0])

    return text_df, service_df, section_df


def extract_section_number(section: str):
    section = str(section).strip()
    numbers = re.findall(r"§\s*(\d+)", section)
    if len(numbers) == 1:
        return numbers[0]
    else:
        return None


def extract_paragraph_number(paragraph: str):
    paragraph = str(paragraph).strip()
    numbers = re.findall(r"\((\d+)\)", paragraph)
    if len(numbers) == 1:
        return numbers[0]
    else:
        return None


def parse_references(references: str, must_contain_section: bool = False):
    references = str(references).strip()
    keywords = ["section", "§", "paragraph", "absatz", "abs.", "sentence", "satz", "number", "nummer"]
    keys = ["section", "section", "paragraph", "paragraph", "paragraph", "sentence", "sentence", "number", "number"]
    if must_contain_section:
        if not re.search(r"§\s*\d+", references, flags=re.IGNORECASE) and not re.search(r"section\s*\d+", references,
                                                                                        flags=re.IGNORECASE):
            return None

    else:
        if not any(re.search(rf"{keyword}\s*\d+", references.lower(), flags=re.IGNORECASE) for keyword in keywords):
            return None

    references = references.split(",")

    parsed_references = []

    for reference in references:
        reference = reference.strip()
        if not any(keyword in reference.lower() for keyword in keywords):
            continue

        if re.search(r"§\s*\d+\s*\(\d+\)", reference, flags=re.IGNORECASE):
            reference = re.sub(r".*(§\s*\d+)\s*\((\d+)\)", r"\1 paragraph \2", reference, flags=re.IGNORECASE)

        # reference = Paragraph 2 number 2
        parsed_reference = {}
        for keyword, key in zip(keywords, keys):
            if keyword in reference.lower():
                # parse the number after the keyword
                numbers = re.findall(rf"{keyword}\s*(\d+)", reference.lower(), flags=re.IGNORECASE)
                if len(numbers) == 1:
                    number = numbers[0]
                else:
                    number = None
                parsed_reference[key] = number
        parsed_references.append(parsed_reference)

    return parsed_references


def complete_reference(parsed_references: List[Dict[str, str]], current_section: str, current_paragraph: str):
    if parsed_references is None:
        return None

    for parsed_reference in parsed_references:
        if len(parsed_reference) == 0:
            continue
        if "section" not in parsed_reference or parsed_reference["section"] is None:
            parsed_reference["section"] = current_section
            if "paragraph" not in parsed_reference or parsed_reference["paragraph"] is None:
                parsed_reference["paragraph"] = current_paragraph
    return parsed_references


def extract_references(text: str, open_ai_model_name: str):
    with get_openai_callback() as cb:
        chat_prompt = ChatPromptTemplate.from_messages(
            [HumanMessagePromptTemplate.from_template(rule_extraction_prompt)])
        chain = LLMChain(llm=ChatOpenAI(temperature=0, model_name=open_ai_model_name), prompt=chat_prompt, verbose=True)
        response = chain.predict(text=text)
        print(response)
        return response, cb.successful_requests, cb.prompt_tokens, cb.completion_tokens, cb.total_cost


def read_in_goz_commentary(filepath):
    text_list = []

    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            text_list.append(text)

    return text_list


def add_service_commentary(service_df, section_df, text_list):
    # if a page starts with a §, add this page to paragraph_commentaries under the respective paragraph. Also all following pages, until there comes another page starting with a §
    service_commentaries = {}
    section_commentaries = {}
    current_service = None
    for page in text_list[:-3]:
        # get first line of page
        first_line = page.split("\n")[0]
        second_line = page.split("\n")[1]

        if re.search(r"GOZ\s*Nr\.?\s*(\d+)", first_line, re.IGNORECASE):
            is_service_commentary = True
        elif re.search(r"GOZ\s*Nr\.?\s*(\d+)", second_line, re.IGNORECASE):
            is_service_commentary = True
            first_line = second_line
        else:
            is_service_commentary = False

        if "Allgemeine Bestimmungen Teil" in second_line:
            # section letter is the letter which comes after "Allgemeine Bestimmungen Teil"
            section_letter = second_line.split("Allgemeine Bestimmungen Teil")[1].strip().split(" ")[0]
            section_commentaries[section_letter] = page
            continue

        # search for GOZ Nr. number, ignore case
        if re.search(r"GOZ\s*Nr\.?\s*(\d+)", first_line, re.IGNORECASE):
            # parse GOZ Nr. number
            goz_number = re.search(r"GOZ\s*Nr\.?\s*(\d+)", first_line, re.IGNORECASE).group(1)

            current_service = goz_number
            service_commentaries[goz_number] = page
        elif current_service is not None:
            service_commentaries[current_service] += "\n\n" + page

    commentary_df = pd.DataFrame.from_dict(service_commentaries, orient="index", columns=["commentary"])
    commentary_df.index = commentary_df.index.astype(int)
    service_df["number"] = service_df["number"].astype(int)
    service_df = service_df.join(commentary_df, on="number")

    section_commentary_df = pd.DataFrame.from_dict(section_commentaries, orient="index", columns=["commentary"])
    section_df = section_df.join(section_commentary_df, on="section_letter")

    return service_df, section_df


def add_text_commentary(text_df, text_list):
    # if a page starts with a §, add this page to paragraph_commentaries under the respective paragraph. Also all following pages, until there comes another page starting with a §
    text_commentaries = {}
    current_section = None
    for page in text_list[:-3]:
        # get first line of page
        first_line = page.split("\n")[0]
        second_line = page.split("\n")[1]

        if re.search(r"§\s*(\d+)", first_line, re.IGNORECASE):
            is_service_commentary = True
        elif re.search(r"§\s*(\d+)", second_line, re.IGNORECASE):
            is_service_commentary = True
            first_line = second_line
        else:
            is_service_commentary = False

        if "Allgemeine Bestimmungen Teil" in second_line:
            break

        # search for GOZ Nr. number, ignore case
        if re.search(r"§\s*(\d+)", first_line, re.IGNORECASE):
            # parse Paragraph. number
            goz_number = re.search(r"§\s*(\d+)", first_line, re.IGNORECASE).group(1)
            if goz_number == current_section:
                text_commentaries[current_section] += "\n\n" + page
            else:
                current_section = goz_number
                text_commentaries[goz_number] = page
        elif current_section is not None:
            text_commentaries[current_section] += "\n\n" + page

    commentary_df = pd.DataFrame.from_dict(text_commentaries, orient="index", columns=["commentary"])
    commentary_df.index = commentary_df.index.astype(int)
    text_df["section_number"] = text_df["section_number"].astype(int)
    text_df = text_df.join(commentary_df, on="section_number")

    return text_df


def translate_text(text, from_language, to_language, open_ai_model_name):
    if not len(text.strip()):
        return text

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                f"You are an expert translator for translating {from_language} into {to_language}."),
            HumanMessagePromptTemplate.from_template(
                "Translated the following text from {from_language} to {to_language}:\n{text}\n\nTranslation:")
        ]
    )
    translation_chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model_name=open_ai_model_name),
        prompt=prompt,
        verbose=True,
    )

    return translation_chain.predict(from_language=from_language, to_language=to_language, text=text)


def translate_parsed_text_segments(text_df, service_df, section_df, from_language, to_language, open_ai_model_name):
    translated_text_df = text_df.copy()
    translated_service_df = service_df.copy()
    translated_section_df = section_df.copy()

    translated_text_df["content"] = translated_text_df["content"].apply(
        lambda x: translate_text(x, from_language, to_language, open_ai_model_name))
    translated_section_df["content"] = translated_section_df["content"].apply(
        lambda x: translate_text(x, from_language, to_language, open_ai_model_name))
    translated_service_df["content"] = translated_service_df["content"].apply(
        lambda x: translate_text(x, from_language, to_language, open_ai_model_name))

    return translated_text_df, translated_service_df, translated_section_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--OPENAI_API_KEY', type=str, required=True)
    parser.add_argument('--file_path', type=str, default='./resources/goz/gebuehrenordnung_fuer_zahnaerzte_2012.pdf')
    parser.add_argument('--output_file_path', type=str, default='./resources/parsed_text')
    parser.add_argument('--use_commentary', action="store_true")
    parser.add_argument('--goz_commentary_file_path', type=str, default='./resources/goz/goz-kommentar-bzaek.pdf')
    parser.add_argument('--open_ai_model_name', type=str, default='gpt-4')
    parser.add_argument('--language', type=str, default='de', choices=['de', 'en'])
    parser.add_argument('--translate_text', type=str, default='None', choices=['english', 'german', "None"])
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ["OPENAI_API_KEY"] = args.OPENAI_API_KEY

    if args.use_commentary and args.translate_text != "None":
        raise NotImplementedError("Translation of the commentary is not implemented yet.")
    if args.use_commentary and args.goz_commentary_file_path is None:
        raise ValueError("You have to specify the path to the goz commentary pdf file.")

    text_df, service_df, section_df = goz_to_sections(args.file_path, args.language)
    text_df, service_df, section_df = preprocess_text_segments(text_df, service_df, section_df, open_ai_model_name=args.open_ai_model_name)

    # add commentary
    if args.use_commentary:
        goz_commentary_pages = read_in_goz_commentary(args.goz_commentary_file_path)
        text_df = add_text_commentary(text_df, goz_commentary_pages)
        service_df, section_df = add_service_commentary(service_df, section_df, goz_commentary_pages)

    if not os.path.exists(args.output_file_path):
        os.makedirs(args.output_file_path)
    if not os.path.exists(f"{args.output_file_path}/{args.language}"):
        os.makedirs(f"{args.output_file_path}/{args.language}")
    text_df.to_csv(f"{args.output_file_path}/{args.language}/text_df.csv", index=False)
    service_df.to_csv(f"{args.output_file_path}/{args.language}/service_df.csv", index=False)
    section_df.to_csv(f"{args.output_file_path}/{args.language}/section_df.csv", index=False)

    if args.translate_text != "None":
        translated_text_df, translated_service_df, translated_section_df = translate_parsed_text_segments(text_df,
                                                                                                          service_df,
                                                                                                          section_df,
                                                                                                          args.language,
                                                                                                          args.translate_text,
                                                                                                          args.open_ai_model_name)
        if not os.path.exists(f"{args.output_file_path}/{args.translate_text}"):
            os.makedirs(f"{args.output_file_path}/{args.translate_text}")
        translated_text_df.to_csv(f"{args.output_file_path}/{args.translate_text}/text_df.csv", index=False)
        translated_service_df.to_csv(f"{args.output_file_path}/{args.translate_text}/service_df.csv", index=False)
        translated_section_df.to_csv(f"{args.output_file_path}/{args.translate_text}/section_df.csv", index=False)
