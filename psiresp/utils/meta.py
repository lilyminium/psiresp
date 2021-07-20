from collections import defaultdict
from typing import Dict, List
import textwrap
import re

INDENT = " " * 4
MERGED_SECTIONS = ["Parameters", "Attributes"]
SECTION_PATTERN = re.compile("(\n\s*\w+\s*\n\s*[-]+\s*\n)")
FIELD_PATTERN = re.compile("(?P<varname>\w+)\s*:\s*(?P<vartype>.*)")


def split_docstring_into_parts(docstring: str) -> Dict[str, Dict[str, List[str]]]:
    """Split docstring around headings"""
    chunks = re.split(SECTION_PATTERN, docstring)[1:]
    parts = {}
    while chunks:
        raw_heading = chunks.pop(0).strip("\n")
        heading = raw_heading.split()[0].capitalize()
        if heading not in MERGED_SECTIONS:
            parts[heading] = f"{raw_heading}\n{chunks.pop(0)}"
            continue

        body = textwrap.dedent(chunks.pop(0))
        current_body = []
        current_field, current_type = None, ""
        fields = {}

        def update_fields():
            if current_field:
                current_body.insert(0, f"{current_field} : {current_type}")
                fields[current_field] = current_body

        for line in body.splitlines():
            if re.match("\s", line):
                current_body.append(line.rstrip())
            else:
                update_fields()
                current_body = []
                match = re.match(FIELD_PATTERN, line)
                if match:
                    current_field = match.group("varname")
                    current_type = match.group("vartype")
                else:
                    current_field, current_type = None, ""
        update_fields()
        parts[heading] = fields
    return parts


def get_cls_docstring_sections(cls) -> Dict[str, dict]:
    """Create docstring from instantiated class"""
    base_parts = split_docstring_into_parts(cls.__doc__)
    schema_parts = schema_to_docstring_sections(cls.__fields__)
    return merge_sections(base_parts, schema_parts)


def merge_sections(*args) -> Dict[str, dict]:
    """Merge dictionaries of dictionaries.

    The arguments should be ordered from *highest to lowest* priority.
    That is, if the same key is present in a value of the first dictionary
    and the second dictionary, the result will return the value in the first
    dictionary.

    Parameters
    ----------
    *args: dictionaries

    Returns
    -------
    dict
    """
    sections = {}
    for heading in MERGED_SECTIONS:
        field = {}
        for arg in args[::-1]:
            field.update(arg.get(heading, {}))
        sections[heading] = field
    if not args:
        return sections
    for key, value in args[0].items():
        if key not in MERGED_SECTIONS:
            sections[key] = value
    return sections


def create_docstring_from_sections(docstring: str,
                                   doc_sections: List[Dict[str, dict]],
                                   order_first: List[str] = []) -> str:
    """Create a new docstring from the original,
    as well as specifications of the Parameters and Attributes sections.
    """
    doc_parts = split_docstring_into_parts(docstring)
    doc_intro = re.split(SECTION_PATTERN, docstring, maxsplit=1)[0]
    merged = merge_sections(doc_parts, *doc_sections)
    print(docstring)
    sections = [doc_intro]

    for heading in MERGED_SECTIONS:
        fields = merged.pop(heading, {})
        if not fields:
            continue
        section_lines = [heading, "-" * len(heading)]
        for name in order_first:
            section_lines.extend(fields.pop(name, []))
        for value in fields.values():
            section_lines.extend(value)
        block = "\n".join(section_lines)
        sections.append(textwrap.indent(block, INDENT))

    sections.extend(list(merged.values()))
    return "\n\n".join(sections)


def schema_to_docstring_sections(cls_fields):
    from pydantic.fields import ModelField

    sections = {name: {} for name in MERGED_SECTIONS}
    for name, field in cls_fields.items():
        if isinstance(field, ModelField) and field.field_info.description:
            ftype = field.outer_type_.__name__
            desc = textwrap.indent(field.field_info.description, INDENT)
            for section_fields in sections.values():
                section_fields[name] = [f"{name} : {ftype}", desc]
    return sections
