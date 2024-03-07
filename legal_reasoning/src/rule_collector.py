import argparse
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='./resources/parsed_text/de')
    parser.add_argument('--output_file_path', type=str, default='temp.pl')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    service_df = pd.read_csv(os.path.join(args.file_path, "service_df.csv"))
    text_df = pd.read_csv(os.path.join(args.file_path, "text_df.csv"))
    section_df = pd.read_csv(os.path.join(args.file_path, "section_df.csv"))

    code = service_df["parsed_code"].tolist()
    code += text_df["parsed_code"].tolist()
    code += section_df["parsed_code"].tolist()
    code = [x for x in code if not str(x) == "nan"]
    code = "\n".join(code)

    with open(args.output_file_path, 'w') as f:
        f.write(code)
