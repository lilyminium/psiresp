

def split_docstring_into_parts(docstring: str) -> Dict[str, List[str]]:
    """Split docstring around headings"""
    parts = defaultdict(list)
    heading_pattern = "[ ]{4}[A-Z][a-z]+\s*\n[ ]{4}[-]{4}[-]+\s*\n"
    directive_pattern = "[ ]{4}\.\. [a-z]+:: .+\n"
    pattern = re.compile("(" + heading_pattern + "|" + directive_pattern + ")")
    sections = re.split(pattern, docstring)
    parts["base"] = sections.pop(0)
    while sections:
        heading_match, section = sections[:2]
        sub_pattern = "([A-Z][a-z]+|[ ]{4}\.\. [a-z]+:: .+\n)"
        heading = re.search(sub_pattern, heading_match).groups()[0]
        section = heading_match + section
        parts[heading] = section.split("\n")
        sections = sections[2:]
    return parts


def join_split_docstring(parts: Dict[str, List[str]]) -> str:
    """Join split docstring back into one string"""
    docstring = parts.pop("base", "")
    headings = ("Parameters", "Attributes", "Examples")
    for heading in headings:
        section = parts.pop(heading, [])
        docstring += "\n".join(section)
    for section in parts.values():
        docstring += "\n".join(section)
    return docstring


def extend_docstring_with_base(docstring: str, base_class: type) -> str:
    """Extend docstring with the parameters in `base_class`"""
    doc_parts = split_docstring_into_parts(docstring)
    base_parts = split_docstring_into_parts(base_class.__doc__)
    headings = ("Parameters", "Attributes", "Examples")
    for k in headings:
        if k in base_parts:
            section = base_parts.pop(k)
            if doc_parts.get(k):
                section = section[2:]
            doc_parts[k].extend(section)

    for k, lines in base_parts.items():
        if k != "base" and k in doc_parts:
            doc_parts[k].extend(lines[2:])

    return join_split_docstring(doc_parts)
