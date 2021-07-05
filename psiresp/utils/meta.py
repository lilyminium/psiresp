from collections import defaultdict
from typing import Dict, List
import re


def split_docstring_into_parts(docstring: str) -> Dict[str, List[str]]:
    """Split docstring around headings"""
    parts = defaultdict(list)
    heading_pattern = "[ ]{4}[ ]*[A-Z][a-z]+\s*\n[ ]{4}[ ]*[-]{4}[-]+\s*\n"
    # directive_pattern = "[ ]{4}[ ]*\.\. [a-z]+::.*\n"
    # pattern = re.compile("(" + heading_pattern + "|" + directive_pattern + ")")
    pattern = re.compile("(" + heading_pattern + ")")
    sections = re.split(pattern, docstring)
    parts[""] = sections.pop(0).rstrip("\n")
    while sections:
        heading_match, section = sections[:2]
        # sub_pattern = "([A-Z][a-z]+|[ ]{4}[ ]*\.\. [a-z]+::.*\n)"
        sub_pattern = "([A-Z][a-z]+)"
        heading = re.search(sub_pattern, heading_match).groups()[0]
        section = heading_match + section
        parts[heading] = section.rstrip("\n").split("\n")

        sections = sections[2:]

    return parts


def join_split_docstring(parts: Dict[str, List[str]]) -> str:
    """Join split docstring back into one string"""
    lines = [parts.pop("", "")]
    headings = ("Parameters", "Attributes", "Examples")
    for heading in headings:
        section = parts.pop(heading, [])
        lines.append("\n".join([x for x in section if x.strip()]))
    for section in parts.values():
        lines.append("\n".join(section))
    docstring = "\n\n".join(lines) + "\n\n"
    return docstring


def extend_docstring_with_base(docstring: str, base_class: type) -> str:
    """Extend docstring with the parameters in `base_class`"""
    doc_parts = split_docstring_into_parts(docstring)
    base_parts = split_docstring_into_parts(base_class.__doc__)
    headings = ("Parameters", "Attributes")
    for k in headings:
        if k in base_parts:
            section = base_parts.pop(k)
            if doc_parts.get(k):
                section = section[2:]
            doc_parts[k].extend([x for x in section if x not in doc_parts[k]])

    # for k, lines in base_parts.items():
    #     if k != "" and k in doc_parts:
    #         doc_parts[k].extend(lines[2:])

    return join_split_docstring(doc_parts)
