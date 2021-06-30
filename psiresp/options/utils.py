import pandas as pd



def add_option_to_init(option_name: str, default: Any) -> Callable:
    """Helper function to add option keywords to __init__.
    
    This makes no attempt to alter the docstring, assuming that the
    user will instead `help(cls)`.

    Parameters
    ----------
    option_name: str
        keyword to add to __init__
    default: Any
        default argument. For containers like lists, should be the
        type to create a new empty default

    Returns
    -------
    decorated function

    """
    fields = default.__dataclass_fields__.keys()
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if isinstance(default, type):
                default = default()
            option_val = kwargs.pop(option_name, default)
            opt_specific = {k: kwargs.pop(k) for k in fields if k in kwargs}
            func(self, *args, **kwargs)
            setattr(self, option_name, option_val)
            for k, v in opt_specific.items():
                setattr(self, k, v)
        return wrapper
    return decorator


def create_option_property(field_name: str,
                           option_name: str,
                           docstrings: Dict[str, str] = {},
                           ) -> property:
    """Helper function to add a property for an option field name"""
    def getter(self):
        return getattr(self, option_name)[field_name]
    
    def setter(self, value):
        options = getattr(self, option_name)
        options[field_name] = value
    
    return property(getter, setter, None, docstrings.get(field_name, ""))


def split_docstring_around_parameters(text: str) -> Tuple[str, List[str], str]:
    """Split docstring into three segments around Parameters.

    Parameters
    ----------
    text: str
        The docstring
    
    Returns
    -------
    before_parameters: str
        The string before the Parameters section
    parameter_list: list of strings
        The parameters section, where each item in the list is a line
    after_parameters: str
        The string after the Parameters section
    
    """
    doclines = text.split("\n")
    before_param = []
    param = []
    after_param = []
    lists = [before_param, param, after_param]
    ticker = 0
    for i in range(len(doclines) - 1):
        first = doclines[i]
        second = doclines[i + 1]
        if "----" in second:
            if "Parameters" in first:
                ticker = 1
            elif ticker:
                ticker = 2
        lists[ticker].append(first)
    lists[ticker].append(doclines[-1])
    lists[0] = "\n".join(lists[0])
    lists[-1] = "\n".join(lists[-1])
    return tuple(lists)


def is_further_indented(line: str, initial_indent: int) -> bool:
    """Returns whether a line is further indented than the `initial_indent`.

    This assumes that indentation is with spaces rather than tabs.
    
    Parameters
    ----------
    line: str
        The line
    initial_indent: int
        The number of spaces

    Returns
    -------
    is_indented: bool
        Whether the line has more spaces at the start than `initial_indent`
    """
    new_indent = len(line) - len(line.lstrip(" "))
    return new_indent > initial_indent


def get_parameter_docstrings(obj: Any) -> Dict[str, str]:
    """Get docstrings for parameters by parsing the `obj` docstring

    It first looks for a "Parameters" section in the docstring,
    and then associates each keyword with the base level of indentation
    with the following lines that are further indented.

    Parameters
    ----------
    obj: Any
        any Python object with a `__doc__` attribute
    
    Returns
    -------
    parameter_docs: dict of {str: str}

    """
    parameter_section = split_docstring_around_parameters(obj.__doc__)[1]
    try:
        first_line = parameter_section[0]
    except IndexError:
        return {}
    initial_indent = len(first_line) - len(first_line.lstrip(" "))

    parameter_docs = defaultdict(list)

    key = None
    for line in parameter_section:
        if is_further_indented(line, initial_indent) and key:
            parameter_docs[key].append(line)
        else:
            key = line.strip().split(":")[0]
    docstrings = {}
    for k, vs in parameter_docs.items():
        docstrings[k] = " ".join([v.strip() for v in vs])
    return docstrings


def extend_new_class_parameters(base_class: Type, params: List[str]):
    """Extend `params` docstring list with the parameters in `base_class`
    
    This is done in-place.
    """
    if not params:
        params.extend(["Parameters", "----------"])
    base_doc_params = split_docstring_around_parameters(base_class.__doc__)[1]
    if "Parameters" in base_doc_params[0]:
        base_doc_params = base_doc_params[2:]
    params.extend(base_doc_params)
