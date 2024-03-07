from typing import List, Union

import networkx as nx

import graph_generation
import re


def initialize_undefined_predicates(code: str):
    undefined_predicates = get_undefined_predicates(code)

    additional_predicates = ""
    for undefined_predicate in undefined_predicates:
        # replace the /number with number dummy arguments, e.g. difficult_service/2 -> difficult_service(Argument1, Argument2)
        args_count = int(re.findall(r'\/(\d+)', undefined_predicate)[0])
        undefined_predicate = re.sub(r'\/(\d+)', r'', undefined_predicate)
        undefined_predicate += "(" + ",".join([f"argument{i}" for i in range(1, args_count + 1)]) + ") :- false."
        additional_predicates += "\n" + undefined_predicate

    code = code + "\n" + additional_predicates
    return code


def get_undefined_predicates(code):
    rule_attributes = graph_generation.RuleAttributes()
    G = graph_generation.read_rules(code, "./resources/temp.html", rule_attributes)
    undefined_predicates = graph_generation.get_undefined_predicates(G, rule_attributes)
    print(undefined_predicates)
    return undefined_predicates


def find_parent_querries(code: str, querries: List[str]) -> List[str]:
    rule_attributes = graph_generation.RuleAttributes()
    G = graph_generation.read_rules(code, "./resources/temp.html", rule_attributes)
    query_predicates = [graph_generation.get_function_name(parsed_query, rule_attributes) for parsed_query in querries]
    # check if one predicate is the child of another predicate in G
    parent_queries = querries.copy()
    for parsed_query, query_predicate in zip(querries, query_predicates):
        for other_query_predicate in query_predicates:
            if query_predicate == other_query_predicate:
                continue
            if nx.has_path(G, query_predicate, other_query_predicate):
                if parsed_query in parent_queries:
                    parent_queries.remove(parsed_query)

    return parent_queries


def find_root_predicates(code: str) -> List[str]:
    rule_attributes = graph_generation.RuleAttributes()
    G = graph_generation.read_rules(code, "temp.html", rule_attributes)
    root_nodes = [node for (node, out_degree), (_, in_degree) in zip(G.out_degree(), G.in_degree()) if
                  out_degree == 0 and in_degree > 0]
    return root_nodes


def correct_negated_facts(code: str) -> str:
    predicates = [re.sub(r'\\\+(\S*?\(.*?\))\.', r'\1 :- false.', predicate) for predicate in code.split("\n")]
    predicates = "\n".join(predicates)
    return predicates


def parse_prolog_output(output: str, query: str) -> Union[bool, str]:
    if f"ERROR: -g {query}: false" in output:
        return False

    if "error" in output.lower():
        return "Error"

    last_output_line = ""
    i = -1
    while not last_output_line.strip() and i >= -len(output.split("\n")):
        last_output_line = output.split("\n")[i]
        i -= 1

    if "warn" not in last_output_line.lower() and last_output_line.strip():
        return last_output_line

    return True


def replace_not_with_not_plus(prolog_program):
    return re.sub(r'not\((.*?)\)', r'\\+(\1)', prolog_program)