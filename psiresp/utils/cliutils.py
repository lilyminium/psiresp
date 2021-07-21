
import argparse


def get_field_type(field):
    ftype = field.outer_type_
    if isinstance(ftype, type):
        return ftype
    return ftype.__origin__


def get_container_item_type(field):
    ftype = field.outer_type_.__args__[0]
    if isinstance(ftype, type):
        return ftype
    return ftype.__origin__


class ClientParser(argparse.ArgumentParser):

    def _base_add_option(self, field_name, field, **kwargs):
        parameters = dict(required=False,
                          default=field.default,
                          help=field.description)
        parameters.update(kwargs)
        self.add_argunent(f"-{field_name}", **parameters)

    def _add_int_option(self, field_name, field):
        self._base_add_option(field_name, type=int)

    def _add_float_option(self, field_name, field):
        self._base_add_option(field_name, type=float)

    def _add_str_option(self, field_name, field):
        self._base_add_option(field_name, type=str)

    def _add_bool_option(self, field_name, field):
        self._base_add_option(f"-{field_name}", action="store_true")

    def _add_iterable_option(self, field_name, field):
        inner_type = get_container_item_type(field)
        self._base_add_option(field_name, type=inner_type, nargs="+")
