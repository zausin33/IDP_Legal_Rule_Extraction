#!/bin/bash

# Function to prompt for input and read the value, using a default if enter is pressed
prompt_for_input() {
    read -p "$1 ($2): " value
    if [ -z "$value" ]; then
        value=$2
    fi
    echo $value
}

# Function to ask for yes/no and return flag accordingly
prompt_for_yes_no() {
    while true; do
        read -p "$1 (yes/no): " yn
        case $yn in
            [Yy]* ) echo "--use_commentary"; break;;
            [Nn]* ) echo ""; break;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

# Check if SWI-Prolog is installed
if ! command -v swipl &> /dev/null
then
    echo "SWI-Prolog is not installed. Please install it to continue."
    exit 1
fi

cd ./src

# Choose the operation mode
echo "Select the operation mode:"
echo "1 - Run Rule Translation Pipeline"
echo "2 - Check an Invoice based on the Rules"
read -p "Enter your choice (1 or 2): " operation_mode

# Conditional execution based on the chosen operation mode
if [ "$operation_mode" == "1" ]; then
    OPENAI_API_KEY=$(prompt_for_input "Enter your OpenAI API Key" "")
    GOZ_PATH=$(prompt_for_input "Enter the path of GOZ file" "./resources/goz/gebuehrenordnung_fuer_zahnaerzte_2012.pdf")
    OUTPUT_PATH_1=$(prompt_for_input "Enter the preprocessing output file path" "./resources/parsed_text_new")
    OUTPUT_PATH_2=$(prompt_for_input "Enter the rule extraction output file path" "./resources/parsed_text_new/parsed_rules")
    OUTPUT_PATH_3=$(prompt_for_input "Enter the rule translation file path" "./resources/results_new")
    USE_COMMENTARY=$(prompt_for_yes_no "Do you want to use the goz commentary?")
    OPEN_AI_MODEL_PREPROCESSING=$(prompt_for_input "Enter the OpenAI model for preprocessing" "gpt-4")
    OPEN_AI_MODEL_RULE_EXTRACTION=$(prompt_for_input "Enter the OpenAI model for rule extraction" "gpt-4")
    OPEN_AI_MODEL_RULE_TRANSLATION=$(prompt_for_input "Enter the OpenAI model for rule translation" "gpt-4")

    # Check if all specified files are present
    if [ ! -f "$GOZ_PATH" ]; then
        echo "The GoZ file is missing. Please check the file path."
        exit 1
    fi

    # Run Rule Translation Pipeline
    echo "Running Rule Translation Pipeline..."

    # Step 1: Text Preprocessing
    echo "Running text preprocessing..."
    python ./text_preprocessing.py --OPENAI_API_KEY="$OPENAI_API_KEY" --file_path="$GOZ_PATH" --output_file_path="$OUTPUT_PATH_1" --open_ai_model_name="$OPEN_AI_MODEL_PREPROCESSING" $USE_COMMENTARY

    # Step 2: Rule Extraction
    echo "Running rule extraction..."
    python ./rule_extraction.py --OPENAI_API_KEY="$OPENAI_API_KEY" --file_path="$OUTPUT_PATH_1/de" --output_file_path="$OUTPUT_PATH_2" --open_ai_model_name="$OPEN_AI_MODEL_RULE_EXTRACTION" $USE_COMMENTARY

    # Step 3: Rule Translation
    echo "Running rule translation..."
    python ./rule_translation.py --OPENAI_API_KEY="$OPENAI_API_KEY" --file_path="$OUTPUT_PATH_2" --output_file_path="$OUTPUT_PATH_3" --open_ai_model_name="$OPEN_AI_MODEL_RULE_TRANSLATION" $USE_COMMENTARY

elif [ "$operation_mode" == "2" ]; then
    OUTPUT_PATH_3=$(prompt_for_input "Enter the rule translation file path" "./resources/results_new")
    INVOICE_JSON_PATH=$(prompt_for_input "Enter the path of the invoice JSON file" "./resources/evaluation_goz/invoices/example_invoice.json")
    INVOICE_PROLOG_OUTPUT_FILE=$(prompt_for_input "Enter the Invoice Prolog output file path" "./invoice.pl")
    PROLOG_RULES_FILE=$(prompt_for_input "Enter the Prolog rules file path" "./temp.pl")

    if [ ! -f "$INVOICE_JSON_PATH" ]; then
        echo "The Invoice Json file is missing. Please check the file path."
        exit 1
    fi

    if [ ! -f "$OUTPUT_PATH_3/service_df.csv" ]; then
        echo "The rule translation file path is not valid. Please check the file path."
        exit 1
    fi

    # Check Invoice based on the Rules
    echo "Checking an Invoice based on the Rules..."

    # Step 4: Rule Collector
    echo "Collecting all translated rules..."
    python ./rule_collector.py --file_path="$OUTPUT_PATH_3" --output_file_path="$PROLOG_RULES_FILE"

    # Step 5: Convert JSON to Prolog
    echo "Converting JSON invoice to Prolog..."
    python ./invoice_json_to_prolog.py --invoice_json_file="$INVOICE_JSON_PATH" --prolog_output_file="$INVOICE_PROLOG_OUTPUT_FILE"

    # Step 6: Test Prolog Rules
    echo "Testing Prolog rules..."
    swipl -g "consult('$PROLOG_RULES_FILE')" -g "consult('$INVOICE_PROLOG_OUTPUT_FILE')" -g "example_invoice(X), is_invoice_invalid(X)." -t halt

else
    echo "Invalid choice. Exiting."
    exit 1
fi

echo "Pipeline execution completed."

echo "Press enter to exit"
read
