import array
import json
from base64 import b64decode, b64encode
from collections.abc import MutableMapping
from enum import Enum, unique
from numbers import Number
from typing import AbstractSet, List, Tuple, Sequence, Dict, Optional, Any, Union

import numpy as np
from pydicom import Dataset
from pydicom.datadict import tag_for_keyword, dictionary_VR
from pydicom.tag import TagType, Tag

__all__ = ["JsonDataset", "JsonVR", "parse_json_value", "to_json_tag", "json_tag_for_keyword"]


def to_json_tag(key: TagType) -> str:
    """Convert a DICOM tag to its corresponding hex value.

    Args:
        key (TagType): The DICOM tag to convert.
    """
    tag = Tag(key)
    return f"{tag.group:04x}{tag.element:04x}".upper()


def json_tag_for_keyword(keyword: str) -> Optional[str]:
    """Convert a DICOM tag's keyword to its corresponding hex value.

    Args:
        keyword (str): The DICOM tag's keyword to convert.
    """
    tag = tag_for_keyword(keyword)
    if tag is None:
        return None

    return to_json_tag(tag)

def parse_json_value(elem: Dict, unpack: bool = True) -> Any:
    """Parse a JSON value into its corresponding Python type.

    Args:
        elem (Dict): The JSON value to parse.
        unpack (bool, optional): Whether to unpack the value. Defaults to True.
    """
    if not "vr" in elem:
        raise ValueError("Element does not have a VR.")

    vr = JsonVR(elem["vr"])

    # SQ allows for nested DICOM+JSON objects
    if vr == JsonVR.SQ and "Value" in elem:
        return [JsonDataset(item) for item in elem["Value"]]

    # PN is abstracted similarly as in pydicom
    elif vr == JsonVR.PN:
        if "Value" in elem and isinstance(elem["Value"], list) and len(elem["Value"]) > 0:
            if "Alphabetic" in elem["Value"][0]:
                return elem["Value"][0]["Alphabetic"]

    elif vr.is_inline_binary_type() and "InlineBinary" in elem:
        buffer = b64decode(elem["InlineBinary"])
        if vr == JsonVR.OW:
            return array.array("H", buffer).tolist()
        elif vr == JsonVR.OD:
            return array.array("d", buffer).tolist()
        elif vr == JsonVR.OF:
            return array.array("f", buffer).tolist()
        return buffer

    elif "BulkDataURI" in elem:
        return elem["BulkDataURI"]

    elif "Value" in elem:
        if unpack and isinstance(elem["Value"], list) and len(elem["Value"]) == 1:
            return elem["Value"][0]
        return elem["Value"]

    return None


def _pack_value(value: Any) -> List[Any]:
    if not isinstance(value, List):
        value = [value]
    return value


class JsonDataset:
    """Performant partial implementation of the pydicom.FileDataset interface.

    This class is a wrapper around a DICOM+JSON dict that acts as pydicom.FileDataset. It supports
    accessing attributes by DICOM tags/keywords.

    Examples:
    >>> # accessing and writing attributes
    >>> ds = JsonDataset(dcm)
    >>> ds.RescaleIntercept
    >>> ds[0x00281052]
    >>> ds["PatientName"] = "SAMPLE-123"

    >>> # set private tag, with an InlineBinary value
    >>> ds.set_json_attr(0x13370000, "OB", b"123")

    Args:
        dataset (Union[Dict, Dataset]): A DICOM+JSON dict or pydicom.Dataset.

    Raises:
        TypeError: If the dataset is not a dict or pydicom.Dataset.
    """

    def __init__(self, dataset: Optional[Union[Dict, Dataset]] = {}):
        super().__init__()

        if isinstance(dataset, Dataset):
            self._dict = dataset.to_json_dict()
        elif isinstance(dataset, dict):
            self._dict: MutableMapping[str, Dict[str, Any]] = dataset
        else:
            raise TypeError("Expected Dataset or dict.")

    def __array__(self) -> np.ndarray:
        """Return the dataset as a numpy array."""
        return np.asarray(self._dict)

    def __contains__(self, key: TagType) -> bool:
        """Check if the dataset contains a given DICOM tag/keyword.

        Args:
            key (TagType): The DICOM tag/keyword to check for.
        """
        return to_json_tag(key) in self._dict

    def copy(self) -> "JsonDataset":
        """Return a copy of the dataset."""
        return JsonDataset(self._dict.copy())

    def __delattr__(self, name: str) -> None:
        """Delete an attribute from the dataset.

        Args:
            name (str): The name of the attribute to delete on the JSON dict or the Dataset.

        Raises:
            AttributeError: If the attribute does not exist on either the Dataset or the JSON dict.
        """
        json_tag = json_tag_for_keyword(name)
        if json_tag is not None and json_tag in self._dict:
            del self._dict[json_tag]
        elif name in self.__dict__:
            del self.__dict__[name]
        else:
            raise AttributeError(name)

    def __delitem__(self, key: TagType):
        """Delete an item from the dataset.

        Args:
            key (TagType): The DICOM tag/keyword to delete.
        """
        json_tag = to_json_tag(key)
        del self._dict[json_tag]

    def get(self, key: TagType, default: Optional[Any] = None) -> Any:
        """Get an attribute from the dataset.

        Args:
            key (TagType): The DICOM tag/keyword to get.
            default (Any, optional): The default value to return if the key does not exists.
        """
        if isinstance(key, str):
            try:
                return self.__getattr__(key)
            except AttributeError:
                return default

        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def get_json_attr(self, key: TagType) -> Any:
        """Get an attribute from the dataset as a DICOM+JSON dict.

        Args:
            key (TagType): The DICOM tag/keyword to get.
        """
        json_tag = to_json_tag(key)
        return self._dict[json_tag]

    def items(self) -> AbstractSet[Tuple[str, Any]]:
        """Return an iterator over the dataset's items."""
        return list(zip(self.keys(), self.values()))

    def keys(self) -> AbstractSet[str]:
        """Return an iterator over the dataset's keys."""
        return self._dict.keys()

    def values(self) -> Sequence[Any]:
        """Return an iterator over the dataset's values."""
        return [parse_json_value(elem) for elem in self._dict.values()]

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the dataset.

        Args:
            name (str): The name of the attribute to get.
        """
        if json_tag_for_keyword(name) is not None:
            return parse_json_value(self.get_json_attr(name))

        return object.__getattribute__(self, name)

    def __getitem__(self, key: TagType) -> Any:
        """Get an item from the dataset.

        Args:
            key (TagType): The DICOM tag/keyword to get.
        """
        return parse_json_value(self.get_json_attr(key))

    def __iter__(self):
        """Return an iterator over the dataset."""
        json_tags = sorted(self._dict.keys())
        for json_tag in json_tags:
            yield self[int(json_tag, 16)]

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self._dict)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute on the dataset.

        Args:
            name (str): The name of the attribute to set.
            value (Any): The value to set.
        """
        if json_tag_for_keyword(name) is not None:
            self.set_json_attr(name, value=value)
        else:
            object.__setattr__(self, name, value)

    def set_bulk_data_uri(self, key: TagType, uri: str, vr: Optional[str] = None):
        """Set a bulk data URI on the dataset.

        Args:
            key (TagType): The DICOM tag/keyword to set.
            uri (str): The bulk data URI.
            vr (str, optional): The VR of the attribute. Defaults to None.
        """
        json_tag = to_json_tag(key)
        vr = JsonVR.from_key(key) if vr is None else JsonVR(vr)
        if not vr.is_bulk_data_uri_type():
            raise ValueError(f"Invalid BulkDataURI VR: {vr.value}.")

        self._dict[json_tag] = {"vr": vr.value, "BulkDataURI": uri}

    def set_json_attr(
        self,
        key: TagType,
        vr: Optional[str] = None,
        value: Optional[Any] = None,
    ):
        """Set the value of a DICOM+JSON element.

        Args:
            key (TagType): The DICOM tag/keyword to set.
            vr (str, optional): The VR of the attribute. Defaults to None.
            value (Any, optional): The value to set. Defaults to None.
        """
        json_tag = to_json_tag(key)
        if json_tag is None:
            raise ValueError(f"Invalid tag: {key}.")

        vr = JsonVR.from_key(key) if vr is None else JsonVR(vr)

        if value is not None:
            if not vr.is_supported_writable_type(value):
                type_names = [t.__name__ for t in vr.supported_writable_types]
                raise ValueError(f"{vr} value must be one of types: {type_names}.")

            if vr == JsonVR.SQ:
                self._dict[json_tag] = {"vr": vr.value, "Value": _pack_value(value)}
            elif vr == JsonVR.PN:
                self._dict[json_tag] = {"vr": vr.value, "Value": [{"Alphabetic": value}]}
            elif vr.is_inline_binary_type():
                self._dict[json_tag] = {"vr": vr.value, "InlineBinary": b64encode(value)}
            else:
                self._dict[json_tag] = {"vr": vr.value, "Value": _pack_value(value)}
        else:
            self._dict[json_tag] = { "vr": vr.value }

    def __setitem__(self, key: TagType, value: Any):
        """Set an item on the dataset.

        Args:
            key (TagType): The DICOM tag/keyword to set.
            value (Any): The value to set.
        """
        self.set_json_attr(key, value=value)

    def to_dataset(self) -> Dataset:
        """Convert the JsonDataset to a pydicom Dataset object."""
        return Dataset.from_json(self._dict)

    def to_json(self) -> str:
        """Convert the JsonDataset to a JSON string."""
        return json.dumps(self._dict)

    def to_json_dict(self) -> Dict:
        """Convert the JsonDataset to a JSON dictionary."""
        return self._dict

    def __repr__(self):
        """Return a string representation of the dataset."""
        return f"{self.__class__.__name__}(elements={len(self)})"


@unique
class JsonVR(Enum):
    """Enum describing supported VRs in DICOM+JSON.

    Allowed VRs in DICOM+JSON:
    https://dicom.nema.org/dicom/2013/output/chtml/part18/sect_F.2.html
    """

    AE = ("AE", (str,))
    AS = ("AS", (str,))
    AT = ("AT", (str,))
    CS = ("CS", (str,))
    DA = ("DA", (str,))
    DS = ("DS", (Number,))
    DT = ("DT", (str,))
    FD = ("FD", (Number,))
    FL = ("FL", (Number,))
    IS = ("IS", (Number,))
    LO = ("LO", (str,))
    LT = ("LT", (str,))
    OB = ("OB", (bytes, bytearray))
    OD = ("OD", (bytes, bytearray))
    OF = ("OF", (bytes, bytearray))
    OW = ("OW", (bytes, bytearray))
    PN = ("PN", (str,))
    SH = ("SH", (str,))
    SL = ("SL", (Number,))
    SQ = ("SQ", (JsonDataset,))
    SS = ("SS", (Number,))
    ST = ("ST", (str,))
    TM = ("TM", (str,))
    UI = ("UI", (str,))
    UL = ("UL", (Number,))
    UN = ("UN", (bytes, bytearray))
    US = ("US", (Number,))
    UT = ("UT", (str,))

    def __new__(cls, vr: str, supported_writable_types: Any):
        obj = object.__new__(cls)
        obj._value_ = vr
        obj.supported_writable_types = supported_writable_types
        return obj

    def is_inline_binary_type(self) -> bool:
        """Return True if the VR is an inline binary type."""
        return self in [JsonVR.OB, JsonVR.OD, JsonVR.OF, JsonVR.OW, JsonVR.UN]

    def is_bulk_data_uri_type(self) -> bool:
        """Return True if the VR is a bulk data URI type."""
        return self in [
            JsonVR.FL, JsonVR.FD, JsonVR.IS, JsonVR.LT, JsonVR.OB, JsonVR.OD, JsonVR.OF, JsonVR.OW,
            JsonVR.SL, JsonVR.SS, JsonVR.ST, JsonVR.UL, JsonVR.UN, JsonVR.US, JsonVR.UT,
        ]

    def is_supported_writable_type(self, value: Any) -> bool:
        """Return True if the value is a supported type for the VR."""
        if isinstance(value, List):
            if self.is_inline_binary_type() or self.is_bulk_data_uri_type():
                return False
            return all([isinstance(v, self.supported_writable_types) for v in value])

        return isinstance(value, self.supported_writable_types)

    @classmethod
    def from_key(cls, key: TagType) -> "JsonVR":
        """Return the VR for the given DICOM tag/keyword."""
        return cls(dictionary_VR(key).upper())
