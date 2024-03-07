import argparse
import json
from datetime import datetime
from typing import Dict, Any

import pandas as pd


def json_to_prolog(invoice_data: Dict[str, Any], service_df: pd.DataFrame) -> str:
    # Extract and format the invoice date.
    invoice_date = datetime.strptime(invoice_data["_Rechnungsdatum"], '%Y-%m-%d')
    prolog_invoice_date = f"date({invoice_date.year}, {invoice_date.month}, {invoice_date.day})"

    # Extract the invoice amount.
    invoice_amount = invoice_data["_Rechnungsbetrag"]

    # Lists to hold the Prolog representation of services and material costs.
    services_prolog_list = []
    material_costs_prolog_list = []

    # Process each item in the invoice.
    for item in invoice_data["_Positionen"]:
        try:
            item_date = datetime.strptime(item["_Behandlungsdatum"], '%Y-%m-%d')
            prolog_item_date = f"date({item_date.year}, {item_date.month}, {item_date.day})"
        except:
            prolog_item_date = "date('', '', '')"

        item_multiplier = item["_Faktor"]
        item_count = item["_Anzahl"]
        item_charge = item["_Betrag"]
        item_description = item["_Leistungsbeschreibung"]
        item_number = item["_Nr"]
        item_justification = item.get("_Begründung", "")

        # Convert tooth areas to Prolog list format.
        tooth_areas = ", ".join([f"tooth({str(area)[0]}, {str(area)[1]})" for area in item.get("_Zahn", [])])
        prolog_tooth_list = f"[{tooth_areas}]"

        # Depending on the type, create a service or a material cost entry.
        if item["_Typ"] == "Service":
            point_score = service_df[service_df["number"] == int(item_number)]["Punktzahl"].values[0]
            # if item count > 0 write the service count times. When the number of teeth == count,
            # then write the service for each tooth.
            if 1 < item_count == len(item.get("_Zahn", [])):
                prolog_teeth = [f"[tooth({str(area)[0]}, {str(area)[1]})]" for area in item.get("_Zahn", [])]
            else:
                prolog_teeth = [prolog_tooth_list]

            for i, teeth in zip(range(item_count), prolog_teeth):
                service_prolog = f"service({item_number}, {prolog_item_date}, {item_multiplier}, {point_score}, {item_charge}, \'{item_description}\', \'{item_justification}\', {teeth})"
                services_prolog_list.append(service_prolog)

        elif item["_Typ"] == "Materialkosten":
            material_cost_prolog = f"material_cost(\'{item_number}\', {prolog_item_date}, {item_multiplier}, {item_count}, {item_charge}, \'{item_description}\', \'{item_justification}\', {prolog_tooth_list})"
            material_costs_prolog_list.append(material_cost_prolog)

    # Convert the lists to strings.
    list_of_services = ",\n\t\t".join(services_prolog_list)

    list_of_material_costs = ",\n\t\t".join(material_costs_prolog_list)
    list_of_material_costs = f",\n\t[\n\t\t{list_of_material_costs}\n\t]"

    # Combine everything into the final Prolog representation.
    prolog_representation = f"invoice([\n\t\t{list_of_services}\n\t]{list_of_material_costs},\n\tinvoice_date({prolog_invoice_date}),\n\tinvoice_amount({invoice_amount})\n)."

    return prolog_representation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--invoice_json_file', type=str,
                        default='./resources/evaluation_goz/invoices/example_invoice.json')
    parser.add_argument('--prolog_output_file', type=str, default='./invoice.pl')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Load the invoice data.
    with open(args.invoice_json_file, "r", encoding="utf-8") as f:
        invoice_json = json.load(f)

    # Load the service data.
    service_df = pd.read_csv('./resources/evaluation_goz/invoices/service_df.csv')

    # Convert the invoice data to Prolog.
    prolog_invoice = json_to_prolog(invoice_json, service_df)

    # Write the Prolog invoice to a file.
    with open(args.prolog_output_file, "w") as f:
        f.write(f"example_invoice({prolog_invoice[:-1]}).")


if __name__ == "__main__":
    main()
