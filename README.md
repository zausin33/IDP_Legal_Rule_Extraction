# Utilizing Large Language Models for Causal Discovery and Legal Text Interpretation: A Case Study on the German GoZ
This is the source code to "Utilizing Large Language Models for Causal Discovery and Legal Text Interpretation: A Case Study on the German GoZ"
The paper can be found [here](resources%2FReport.pdf) and the presentation [here](resources%2FPresentation.pdf).

The project is split into two parts. The first part is in [causal_discovery](causal_discovery) where preliminary experiments were conducted.
The second and most important part of this project is the Python application in [legal_reasoning](legal_reasoning).

It is a application that uses the OpenAI API to translate and process rules from the German Dentist Fee Schedule (GOZ) into a logical programming language. 
The data is preprocessed and stored in CSV files. The application uses the LLMs (Language Model) from OpenAI to extract rules and translate them into Prolog.

The workflow of the application can be seen in the following image:
![rule_extraction.png](resources%2Frule_extraction.png)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.8 or higher
- pip
- SWI Prolog available via command line
- OpenAI API key

### Installing

1. Clone the repository
2. cd into [legal_reasoning](legal_reasoning)
3. Install the required Python packages with `pip install -r requirements.txt`

### Run
cd into [legal_reasoning](legal_reasoning)

To run the whole translation pipeline, run [run.sh](legal_reasoning%2Frun.sh). The script will prompt you for all necessary parameters.

For executing the pipeline step by step, run the following scripts in order:
1. `python .\text_preprocessing.py --OPENAI_API_KEY="<OPENAI_API_KEY>" --file_path="<PATH_OF_GOZ>" --output_file_path="<OUTPUT_PATH_1>" --use_commentary`
2. `python .\rule_extraction.py --OPENAI_API_KEY="<OPENAI_API_KEY>" --file_path="<OUTPUT_PATH_1>/de" --output_file_path="<OUTPUT_PATH_2>"`
3. `python .\rule_translation.py --OPENAI_API_KEY="<OPENAI_API_KEY>" --file_path="<OUTPUT_PATH_2>" --output_file_path="<OUTPUT_PATH_3>" --use_commentary`
4. For collecting all translated rules in one file, run `python .\rule_collector.py --file_path='<OUTPUT_PATH_3>/<SPECIFIC_FOLDER> --output_file_path=<PROLOG_RULES_FILE>'`
5. For converting a dentist invoice from json to prolog, run `python .\invoice_json_to_prolog.py --invoice_json_file='<PATH_OF_INVOICE_JSON>' --prolog_output_file='<INVOICE_PROLOG_OUTPUT_FILE>'`
A example for a dentist invoice can be found in [example_invoice.json](legal_reasoning%2Fsrc%2Fresources%2Fevaluation_goz%2Finvoices%2Fexample_invoice.json)
6. For testing the prologs rules against the invoice run `swipl -g "consult('<PROLOG_RULES_FILE>')" -g "consult('<INVOICE_PROLOG_OUTPUT_FILE>')" -g "example_invoice(X), is_invoice_invalid(X)." -t halt`

## Authors

* **Jonas Zausinger** - [zausin33](https://github.com/zausin33)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
