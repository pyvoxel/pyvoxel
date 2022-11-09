"""
HTTP I/O.

This module contains HTTP input/output helpers.
"""

import re
import zipfile
from io import BytesIO
from typing import Collection, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import requests
from tqdm.auto import tqdm

from voxel.io.dicom import DicomReader
from voxel.io.format_io import DataReader, ImageDataFormat
from voxel.io.nifti import NiftiReader

__all__ = ["HttpReader"]


_MIME_TYPES_ZIP = [
    "application/zip",
    "application/x-zip-compressed",
    "multipart/x-zip",
    "application/dicom+zip",
]


class HttpReader(DataReader):
    """A class for reading DICOMs from HTTP requests with DICOMweb support.

    Attributes:
        verbose (bool, optional): If ``True``, show loading progress bar.
        block_size (int, optional): Block size for reading data.
        **kwargs: Keyword arguments for :class:`DicomReader`.

    Examples:
        >>> hr = HttpReader()
        >>> hr.read("https://server.com/dicom.zip")
        >>> hr.read("https://server.com/dicom.dcm")
        >>> hr.read("https://server.com/dicom-web/studies/x/series/y")
        >>> hr.close()

        >>> with HttpReader() as hr:
        >>>     hr.session.auth = ("username", "password")
        >>>     hr.read("https://server.com/dicom", params={"x": "y"})
    """

    def __init__(self, verbose: bool = False, block_size: int = 10**6, **kwargs):
        self.verbose = verbose
        self.block_size = block_size
        self.session = requests.Session()
        self.pbar = None
        self.kwargs = kwargs

    def _read_multipart_stream(
        self,
        res: requests.Response,
        content_info: str,
    ) -> List[bytes]:
        """Read multipart stream.

        Args:
            res (requests.Response): Response object.
            content_info (str): Content info.
            pbar (tqdm): Progress bar.
        """
        boundary = _extract_boundary(content_info)
        blob, parts = bytes(), []

        for block in res.iter_content(self.block_size):
            self.pbar.update(len(block))

            blob += block
            while boundary in blob:
                part, blob = blob.split(boundary, maxsplit=1)
                content = _extract_part(part)

                if content is not None:
                    parts.append(content)

        content = _extract_part(blob)
        if content is not None:
            parts.append(content)

        self.pbar.close()
        return parts

    def _read_stream(self, res: requests.Response) -> bytes:
        """Read stream.

        Args:
            res (requests.Response): Response object.
            pbar (tqdm): Progress bar.
        """

        blob = bytes()
        for block in res.iter_content(self.block_size):
            self.pbar.update(len(block))
            blob += block

        self.pbar.close()
        return blob

    def _read_dicom(self, buffers: List[bytes], **kwargs):
        """Read DICOMs from data.

        Args:
            buffers (List[bytes]): List of bytes objects.
            **kwargs: Keyword arguments for :class:`DicomReader`.
        """

        # do not pass verbose to reader, as files are already opened
        dr = DicomReader()
        return dr.read([BytesIO(buffer) for buffer in buffers], **kwargs)

    def _read_nifti(self, buffer: bytes, **kwargs):
        """Read NIfTI from data.

        Args:
            buffer (bytes): Bytes
            **kwargs: Keyword arguments for :class:`NiftiReader`.
        """
        nr = NiftiReader()
        return nr.read(BytesIO(buffer), **kwargs)

    def load(
        self,
        url: str,
        params: Union[Dict, List[Tuple], bytes] = np._NoValue,
        data_format: ImageDataFormat = None,
        verbose: bool = None,
        **kwargs,
    ):
        """Load data from HTTP request.

        Args:
            url (str): URL.
            params (Union[Dict, List[Tuple], bytes], optional): Parameters to send with the request.
            **kwargs: Keyword arguments for :class:`DicomReader`.
        """

        if not _is_valid_url(url):
            raise IOError(f"Invalid URL: {url}.")

        params = params if params != np._NoValue else self.session.params
        with self.session.get(url, params=params, stream=True) as res:
            content_length = res.headers.get("Content-Length", 0)
            content_type = res.headers.get("Content-Type", "application/octet-stream").lower()

            self.pbar = tqdm(
                total=content_length,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                disable=not verbose if verbose is not None else not self.verbose,
            )

            # Mime: multipart/related, expect DICOM
            if content_type.startswith("multipart/related;"):
                _, *content_info = [part.strip() for part in content_type.split(";")]
                parts = self._read_multipart_stream(res, content_info)
                return self._read_dicom(parts, **kwargs)

            # Mime: application/zip, expect DICOM
            if content_type in _MIME_TYPES_ZIP:
                blob = self._read_stream(res)
                z = zipfile.ZipFile(BytesIO(blob))
                parts = [z.read(zinfo) for zinfo in z.infolist() if zinfo.file_size > 0]
                return self._read_dicom(parts, **kwargs)

            # Fallback to single file, expect NiFTI or DICOM
            blob = self._read_stream(res)
            if data_format is None:
                basename = urlparse(url).path
                data_format = ImageDataFormat.get_image_data_format(basename)
            elif isinstance(data_format, str):
                data_format = ImageDataFormat[data_format]

            if data_format == ImageDataFormat.nifti:
                if urlparse(url).path.endswith(".gz"):
                    kwargs = {"compressed": True, **kwargs}
                return self._read_nifti(blob, **kwargs)
            elif data_format == ImageDataFormat.dicom:
                return self._read_dicom([blob], **kwargs)
            else:
                raise IOError(f"Unsupported data format: {data_format}.")

    def close(self):
        """Close the current HTTP session."""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def __serializable_variables__(self) -> Collection[str]:
        return self.__dict__.keys()

    read = load  # pragma: no cover


def _is_valid_url(url: str) -> bool:
    """Check if a string represents a valid URL.

    Args:
        url (str): URL.

    Returns:
        bool: Result of the URL validation.
    """
    regex = re.compile(
        r"^(?:http|ftp)s?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )

    return re.match(regex, url) is not None


def _extract_part(part: bytes) -> Union[bytes, None]:
    """Extract part from multipart stream."""
    if part in [b"", b"--", b"\r\n"] or part.startswith(b"--\r\n"):
        return None

    idx = part.index(b"\r\n\r\n")
    if idx > -1:
        return part[idx + 4 :]

    raise ValueError("Part is not CRLF CRLF terminated.")


def _extract_boundary(content_info: List[str]) -> Optional[bytes]:
    """Extract boundary from content info."""
    for item in content_info:
        if "=" not in item:
            continue

        key, value = item.split("=", maxsplit=1)
        if key.lower() == "boundary":
            return b"--" + value.strip('"').encode("utf-8")

    return None
