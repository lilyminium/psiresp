import abc
import pathlib

from . import psi4utils

class ContainsChildMixin(abc.ABC):

    @property
    @abc.abstractclassmethod
    def _child_class(self):
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def _child_container(self):
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def _child_name_template(self):
        raise NotImplementedError

    def _add_child(self,
                   coordinates: Optional[npt.NDArray] = None,
                   **kwargs):
        counter = len(self._child_container) + 1
        name = self._child_name_template.format(name=self.name,
                                                counter=counter)

        clone = self.psi4mol.clone()
        if coordinates is not None:
            psi4utils.set_psi4mol_geometry(clone, coordinates)
        clone.set_name(name)
        new_kwargs = self.to_dict()
        new_kwargs["name"] = name
        new_kwargs.update(kwargs)
        child = self._child_class(clone, **new_kwargs)
        self._child_container.append(child)
        return child


class ContainsParentMixin(abc.ABC):

    @property
    @abc.abstractmethod
    def _parent(self):
        raise NotImplementedError

    @property
    def path(self):
        stem = pathlib.Path(self._parent.path)
        return stem / self.name

