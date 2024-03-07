import networkx as nx
from pyvis.network import Network
from typing import List, Union, Tuple, Any
import re

SPECIAL_CONSTRUCTS = ["forall(member"]


class RuleAttributes:
    def __init__(self):
        self.main_functions = {}
        self.arithmetic_facts = []
        self.facts = []
        self.negations = []
        self.or_count = 0
        self.and_count = 0
        self.for_counter = 0
        self.if_counter = 0
        self.then_counter = 0
        self.else_counter = 0
        self.negation_counter = 0



def detect_loops(G: nx.DiGraph) -> List[List[str]]:
    """
    Detect loops in the graph using DFS.
    """
    loops = []

    def dfs(node, visited, stack):
        visited[node] = True
        stack.append(node)

        for neighbor in G.successors(node):
            if not visited[neighbor]:
                dfs(neighbor, visited, stack)
            elif neighbor in stack:
                # loop detected
                loop = stack[stack.index(neighbor):]
                loops.append(loop)

        stack.pop()

    visited = {node: False for node in G.nodes()}
    stack = []

    for node in G.nodes():
        if not visited[node]:
            dfs(node, visited, stack)

    # remove fill nodes from loops
    fill_nodes = ["AND_", "OR_", "IF_", "THEN_", "ELSE_", "NOT_"]
    for loop in loops:
        for node in loop:
            if any([node.startswith(fill_node) for fill_node in fill_nodes]):
                loop.remove(node)
        if len(loop) <= 1:
            loops.remove(loop)
    return loops


def get_undefined_predicates(G: nx.DiGraph, rule_attributes: RuleAttributes) -> List[str]:
    fill_nodes = ["AND_", "OR_", "IF_", "THEN_", "ELSE_"]
    return [node for node, in_degree in G.in_degree() if
            in_degree == 0
            and node not in rule_attributes.arithmetic_facts
            and node not in rule_attributes.facts
            and not any([node.startswith(fill_node) for fill_node in fill_nodes])]


def change_source_node(G: nx.DiGraph, old_source: str, new_source: str) -> nx.DiGraph:
    """
    Changes the source node of all edges that have the old source node as source.
    E.g. (old_source -> target1, old_source -> target2) becomes (new_source -> target1, new_source -> target2)
    """
    edges_to_change = [(source, target) for source, target in G.edges() if source == old_source]
    for _, target in edges_to_change:
        G.remove_edge(old_source, target)
        G.add_edge(new_source, target)
    return G


def insert_intermediate_nodes(G: nx.DiGraph, new_intermediate_nodes: List[str], existing_node: str) -> nx.DiGraph:
    """
    Inserts intermediate nodes between the existing node and its source nodes.
    E.g. (source1 -> existing_node, source2 -> existing_node, existing_node ->target1) becomes
    (source1 -> new_intermediate_nodes[0], source2 -> new_intermediate_nodes[0],
    new_intermediate_nodes[0] -> ... -> new_intermediate_nodes[-1], new_intermediate_nodes[-1] -> existing_node,
    existing_node ->target1)
    """
    # replace the existing node with the new intermediate node (so the intermediate nodes gets all children of the existing node)
    G = nx.relabel_nodes(G, {existing_node: new_intermediate_nodes[0]})

    # add the existing node again and give him its old parents
    G.add_node(existing_node)
    G = change_source_node(G, new_intermediate_nodes[0], existing_node)

    for i, node in enumerate(new_intermediate_nodes[1:]):
        G.add_node(node)
        # new_intermediate_nodes[i] is the new_intermediate_nodes.index(node) - 1
        G.add_edge(new_intermediate_nodes[i], node)

    # give the new intermediate node the existing node as parent
    G.add_edge(new_intermediate_nodes[-1], existing_node)
    return G


def get_children(G: nx.DiGraph, node: str) -> List[str]:
    return [source for source, target in G.edges() if target == node]


def get_or_child(G: nx.DiGraph, node: str) -> List[str]:
    return [source for source, target in G.edges() if target == node and source.startswith("OR_")]


def split_at_top_level(s: str, delimiter: str) -> List[str]:
    parts = []
    balance = 0
    current_part = []

    for char in s:
        if char == "(":
            balance += 1
        elif char == ")":
            balance -= 1
        elif char == delimiter and balance == 0:
            parts.append(''.join(current_part))
            current_part = []
            continue
        current_part.append(char)

    if current_part:
        parts.append(''.join(current_part))
    return parts


def is_sourrounded_by_top_level_brackets(s: str) -> bool:
    balance = 0

    if not s.startswith("(") or not s.endswith(")"):
        return False

    s = s[1:]

    for idx, char in enumerate(s):
        if balance < 0:
            return False
        if char == "(":
            balance += 1
        elif char == ")":
            balance -= 1
        # if we arrive at second last character and balance is 0, and the last character is a closing bracket

    return True



def parse_condition(condition: str, rule_attributes: RuleAttributes) -> Union[str, Tuple[Any]]:
    condition = re.sub(r"\.$", "", condition)
    if "->" in condition:
        if_cond, body = condition.split("->", maxsplit=1)
        parts = split_at_top_level(body, ";")
        then_cond, else_cond = ";".join(parts[:-1]), parts[-1]
        nested_if = parse_condition_for_brackets(if_cond, rule_attributes)
        nested_then = parse_condition_for_brackets(then_cond, rule_attributes)
        nested_else = parse_condition_for_brackets(else_cond, rule_attributes)

        if_node = f"IF_{rule_attributes.if_counter}"
        then_node = f"THEN_{rule_attributes.then_counter}"
        else_node = f"ELSE_{rule_attributes.else_counter}"
        rule_attributes.if_counter += 1
        rule_attributes.then_counter += 1
        rule_attributes.else_counter += 1
        return (if_node, nested_if, (then_node, nested_then), (else_node, nested_else))

    if "(" not in condition or ":=" in condition:
        rule_attributes.arithmetic_facts.append(condition)
    if "MINUS" in condition or "PLUS" in condition:
        condition = condition.replace("MINUS", "-")
        condition = condition.replace("PLUS", "+")
        rule_attributes.arithmetic_facts.append(condition)

    return condition


def parse_condition_for_brackets(condition: str, rule_attributes: RuleAttributes) -> Union[str, Tuple[str]]:
    """Parse conditions for brackets and handle special cases."""
    parsed_conditions = []
    condition = condition.replace(" is ", ":=")
    condition = condition.replace(" - ", "MINUS")
    condition = condition.replace(" + ", "PLUS")
    condition = condition.replace(" = ", ":=")
    condition = condition.strip().replace(" ", "")

    # Handle negation
    is_negation = False
    if condition.startswith("\+"):
        is_negation = True
        condition = condition[2:]

    if condition.startswith("forall(member"):
        for_all_regex = r"forall\(member\(([^,]*),([^,]*)\),"
        for_expression = "forall member(" + ", ".join(re.findall(for_all_regex, condition)[0]) + ")_" + str(rule_attributes.for_counter)
        rule_attributes.for_counter += 1
        condition = re.sub(for_all_regex, "", condition)
        condition = re.sub(r"\)\)$", "", condition)
        condition = re.sub(r"^\(", "", condition)
        conditions = split_at_top_level(condition, ",")

        parsed_conditions.extend([for_expression] + [parse_condition_for_brackets(condition, rule_attributes) for condition in conditions])
        return tuple(parsed_conditions)

    if condition.startswith("findall("):
        # parse the findall expression into object, goal, and list
        conditions = condition.replace("findall(", "")
        conditions = re.sub(r"\)$", "", conditions)
        conditions = split_at_top_level(conditions, ",")
        if len(conditions) != 3:
            print("Error: findall expression not parsable")
        object = conditions[0]
        goal = conditions[1]
        list = conditions[2]
        find_all_expression = f"findall_{rule_attributes.for_counter}"
        object_expression = f"OBJECT_{rule_attributes.for_counter}"
        goal_expression = f"GOAL_{rule_attributes.for_counter}"
        list_expression = f"LIST_{rule_attributes.for_counter}"
        rule_attributes.for_counter += 1
        parsed_conditions.extend([find_all_expression]
                                 + [(object_expression, parse_condition_for_brackets(object, rule_attributes))]
                                 + [(goal_expression, parse_condition_for_brackets(goal, rule_attributes))]
                                 + [(list_expression, parse_condition_for_brackets(list, rule_attributes))]
                                 )
        return tuple(parsed_conditions)

    # Handle conditions inside brackets
    if is_sourrounded_by_top_level_brackets(condition):
        condition = re.sub(r"^\(", "", condition)
        condition = re.sub(r"\)$", "", condition)

        if_then_else_parts = split_at_top_level(condition, "->")
        if len(if_then_else_parts) > 1:
            inner_conditions = condition
        else:
            inner_conditions = split_at_top_level(condition, ",")

        if is_negation:
            intermediate_node = "NOT_" + str(rule_attributes.negation_counter)
            rule_attributes.negation_counter += 1
        elif len(inner_conditions) > 1:
            intermediate_node = "AND_" + str(rule_attributes.and_count)
            rule_attributes.and_count += 1

        if len(inner_conditions) > 1:
            parsed_inner_conditions = [parse_condition_for_brackets(inner_condition, rule_attributes) for inner_condition in inner_conditions]
            parsed_conditions.extend([intermediate_node] + parsed_inner_conditions)
            return tuple(parsed_conditions)
        else:
            return parse_condition_for_brackets(condition, rule_attributes)
    # completely normal condition
    else:
        # Check for OR condition outside brackets
        or_conditions = split_at_top_level(condition, ";")
        if len(or_conditions) > 1 and "->" not in or_conditions[0]:
            # or_conditions[0] = re.sub(r"^\(+", "", or_conditions[0]) Why were they needed in the first place?
            # or_conditions[-1] = re.sub(r"\)\)$", ")", or_conditions[-1])
            intermediate_node = "OR_" + str(rule_attributes.or_count)
            rule_attributes.or_count += 1
            parsed_or_conditions = [parse_condition_for_brackets(or_condition, rule_attributes) for or_condition in or_conditions]
            parsed_conditions.extend([intermediate_node] + parsed_or_conditions)
            return tuple(parsed_conditions)

        if is_negation:
            rule_attributes.negations.append(get_function_name(condition, rule_attributes))
        return parse_condition(condition, rule_attributes)


def parse_rule(rule: str, rule_attributes: RuleAttributes, split: str = ":-") -> (str, List[str]):
    # Split the rule into main function and conditions
    rule = re.sub(r"\.$", "", rule)
    parts = rule.replace("'", "").split(split)
    main_function = parts[0].strip().replace(" ", "")

    if len(parts) == 1:
        return main_function, [], []

    conditions = split_at_top_level(parts[1], ",")

    condition_functions = []
    for condition in conditions:
        parsed_cond = parse_condition_for_brackets(condition, rule_attributes)
        condition_functions.append(parsed_cond)

    return main_function, condition_functions


def check_for_same_name_with_other_arguments_in_graph(G, node):
    for other_node in G.nodes():
        if "(" in node and "(" in other_node and node.split("(")[0] == other_node.split("(")[0] and node != other_node:
            print(node, other_node)
            return True
    return False


def get_function_name(node, rule_attributes: RuleAttributes, ignore_arguments=True):
    if ignore_arguments and not node in rule_attributes.arithmetic_facts:
        # split at \w( and take the first part and keep the \w, then count the number of arguments inside the ()
        # and append the count
        # e.g. "member(X, [1,2,3])" -> "member/2"
        try:
            parts = re.split(r"((?<=\w)\()", node)
            node_name = parts[0]
            arguments = "".join(parts[1:])
        except ValueError:
            node_name = node
            arguments = ""
        arguments = re.sub(r"^\(", "", arguments)
        arguments = re.sub(r"\)$", "", arguments)
        arguments = split_at_top_level(arguments, ",")
        count = len(arguments)
        node = node_name + "/" + str(count)
    return node


def add_to_graph(G: nx.DiGraph, main_function: str, condition_functions: List[Union[str, Tuple[Any]]], rule_attributes) -> nx.DiGraph:
    # Modify the node name
    main_function = get_function_name(main_function, rule_attributes)
    # Add main function to graph if not already added
    if main_function not in G:
        G.add_node(main_function)
        rule_attributes.main_functions[main_function] = 1
        check_for_same_name_with_other_arguments_in_graph(G, main_function)
    elif main_function in rule_attributes.main_functions.keys() and rule_attributes.main_functions[main_function] == 1:
        # If main function was already added as other main function, we must express the OR condition
        # Add a new node that represents the OR condition
        or_node = "OR_" + str(rule_attributes.or_count)
        rule_attributes.or_count += 1
        existing_children_of_main_function = get_children(G, main_function)
        if len(existing_children_of_main_function) > 1:
            and_node_1 = "AND_" + str(rule_attributes.and_count)
            rule_attributes.and_count += 1
            G = insert_intermediate_nodes(G, [and_node_1, or_node], main_function)
        else:
            G = insert_intermediate_nodes(G, [or_node], main_function)

        if len(condition_functions) > 1:
            and_node_2 = "AND_" + str(rule_attributes.and_count)
            rule_attributes.and_count += 1
            G.add_node(and_node_2)
            G.add_edge(and_node_2, or_node)

        rule_attributes.main_functions[main_function] += 1
        main_function = and_node_2 if len(condition_functions) > 1 else or_node
    elif main_function in rule_attributes.main_functions.keys() and rule_attributes.main_functions[main_function] > 1:
        # If main function was already added as OR condition, we must express the new OR condition
        or_node = get_or_child(G, main_function)[0]
        rule_attributes.main_functions[main_function] += 1
        if len(condition_functions) > 1:
            and_node = "AND_" + str(rule_attributes.and_count)
            rule_attributes.and_count += 1
            G.add_node(and_node)
            G.add_edge(and_node, or_node)
            main_function = and_node
        else:
            main_function = or_node
    else:
        rule_attributes.main_functions[main_function] = 1

    def process_tuple_condition(condition, main_function):
        G.add_node(condition[0])
        G.add_edge(condition[0], main_function)
        for nested_condition in condition[1:]:
            if isinstance(nested_condition, tuple):
                process_tuple_condition(nested_condition, condition[0])
            else:
                nested_condition = get_function_name(nested_condition, rule_attributes)
                if nested_condition not in G:
                    G.add_node(nested_condition)
                    check_for_same_name_with_other_arguments_in_graph(G, nested_condition)
                G.add_edge(nested_condition, condition[0])

    # Add each condition as a node and create an edge
    for condition in condition_functions:
        if not isinstance(condition, tuple):
            condition = get_function_name(condition, rule_attributes)
            if condition not in G:
                G.add_node(condition)
                check_for_same_name_with_other_arguments_in_graph(G, condition)
            G.add_edge(condition, main_function)
        else:
            process_tuple_condition(condition, main_function)

    return G


def add_fact_to_graph(G: nx.DiGraph, fact: str, rule_attributes: RuleAttributes) -> nx.DiGraph:
    condition_name = get_function_name(fact, rule_attributes)
    if condition_name not in G:
        print(f"Fact {fact} not found in graph")
        return G
    if len(fact) > 50:
        rule_attributes.facts.remove(fact)
        fact = f"fact_{condition_name}"
        rule_attributes.facts.append(fact)

    G.add_node(fact)
    G.add_edge(fact, condition_name)
    return G


def generate_graph(rules: List[str], facts: List[str], output_html_name: str, rule_attributes: RuleAttributes) -> nx.DiGraph:
    # Create an empty directed graph
    G = nx.DiGraph()

    # Parse each rule and add it to the graph
    for rule in rules:
        main_function, condition_functions = parse_rule(rule, rule_attributes)
        G = add_to_graph(G, main_function, condition_functions, rule_attributes)
        condition_functions.append(condition_functions)

    for idx, fact in enumerate(facts):
        fact = fact.replace("'", "")
        fact = re.sub(r"\.$", "", fact)
        rule_attributes.facts.append(fact)
        G = add_fact_to_graph(G, fact, rule_attributes)
        facts[idx] = fact

    nodes_to_remove = []
    for node in G.nodes():
        if not node.strip():
            print("Empty node found")
            nodes_to_remove.append(node)
    for node in nodes_to_remove:
        G.remove_node(node)

    # Draw the graph using pyvis
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")
    nt = Network(notebook=True, height="750px", width="100%", directed=True)
    nt.toggle_physics(False)
    nt.from_nx(G)

    legend = {
        "OR condition": ("green", "square"),
        "Negation": ("red", "dot"),
        "IF condition": ("yellow", "diamond"),
        "THEN Branch": ("#90EE90", "triangle"),
        "ELSE Branch": ("#FFA500", "triangleDown"),
        "Arithmetic fact": ("gray", "dot"),
        "Fact definition": ("black", "dot"),
    }
    conditions = [
        (lambda label: label in rule_attributes.arithmetic_facts, "Arithmetic fact"),
        (lambda label: label.startswith("OR_"), "OR condition"),
        (lambda label: label.startswith("IF_"), "IF condition"),
        (lambda label: label.startswith("THEN_"), "THEN Branch"),
        (lambda label: label.startswith("ELSE_"), "ELSE Branch"),
        (lambda label: label.startswith("NOT_") or label in rule_attributes.negations, "Negation"),
        (lambda label: label in rule_attributes.facts, "Fact definition"),
    ]

    for node, (x, y) in pos.items():
        n = nt.get_node(node)
        n['x'] = x
        n['y'] = y

        for condition, key in conditions:
            if condition(n["label"]):
                n["color"] = legend[key][0]
                n["shape"] = legend[key][1]
                break  # exit the loop once a condition is met

    def add_legend(nt, legend):
        for idx, (description, properties) in enumerate(legend.items()):
            nt.add_node(description, color=properties[0], shape=properties[1], x=0, y=100 * idx)

    add_legend(nt, legend)

    nt.show(output_html_name)

    return G


def read_rules(input_text: str, output_html_name: str, rule_attributes: RuleAttributes) -> nx.DiGraph:
    # Splitting the content into lines for easy parsing
    lines = input_text.strip().split("\n")

    facts = []
    rules = []

    # Parsing the content
    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx]
        rule_clauses = []
        # remove everything after a comment
        if "%" in line:
            line = line.split("%")[0]
        if ":-" in line and not line.startswith("%"):  # This is a rule
            rule_clauses.append(line)
            # append predicates to rules as long as the following lines have an indentation
            while line_idx + 1 < len(lines) and (lines[line_idx + 1].startswith("  ") or lines[line_idx + 1].startswith("\t")):
                if "%" in lines[line_idx + 1]:
                    lines[line_idx + 1] = lines[line_idx + 1].split("%")[0]
                if lines[line_idx + 1].strip():
                    rule_clauses.append(lines[line_idx + 1].strip())
                line_idx += 1

            rules.append(" ".join(rule_clauses))
        elif len(line) > 0 and not line.startswith("  ") and not line.startswith("%"):  # This is a fact
            fact_clauses = [line]
            while line_idx + 1 < len(lines) and (lines[line_idx + 1].startswith("  ") or lines[line_idx + 1].startswith("\t")):
                if "%" in lines[line_idx + 1]:
                    lines[line_idx + 1] = lines[line_idx + 1].split("%")[0]
                if lines[line_idx + 1].strip():
                    fact_clauses.append(lines[line_idx + 1].strip())
                line_idx += 1
            facts.append(" ".join(fact_clauses))

        line_idx += 1

    try:
        return generate_graph(rules, facts, output_html_name, rule_attributes)
    except Exception as e:
        print(e)
        print(input_text)
        raise e




def read_rules_from_file(input_file_name: str, output_html_name: str, rule_attributes: RuleAttributes) -> nx.DiGraph:
    with open(input_file_name, "r") as f:
        prolog_content = f.read()
    return read_rules(prolog_content, output_html_name, rule_attributes)


if __name__ == "__main__":
    rule_attributes = RuleAttributes()
    G = read_rules_from_file("../temp.pl", "../temp.html", rule_attributes)
    loops = detect_loops(G)
    undefined_predicates = get_undefined_predicates(G, rule_attributes)
    print("loops:", loops)
    print("undefined predicates", undefined_predicates)
