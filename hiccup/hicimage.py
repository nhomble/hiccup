import abc
import numpy as np
from .transform import *


class HicImage:
    """
    Represents a HIC image which has dimensions and a compression style
    """

    def __init__(self, rows: int, columns: int, transform: Transform, payload: np.ndarray):
        self._rows = rows
        self._columns = columns
        self._transform = transform
        self._payload = payload

    @staticmethod
    def load_file(file):
        with open(file, 'r') as f:
            header = f.readline()
            content = f.readline()
        header_content = HicImage._parse_header(header)
        return HicImage(
            header_content["rows"],
            header_content["columns"],
            header_content["transform"],
            content
        )

    @staticmethod
    def _parse_header(header):
        header = header.replace("\n", "")
        splits = header.split(" ")
        return {
            "rows": int(splits[0]),
            "columns": int(splits[1]),
            "transform": Transform.from_string(splits[2], splits[3:])
        }

    def _output_path(self, path: str):
        if path.endswith(".hic"):
            return path
        return path + ".hic"

    def _format_header(self):
        return "{} {} {} {}".format(self._rows, self._columns, self.transform.style(),
                                    self._transform.format_parameters)

    def _format_payload(self):
        return np.array2string(self._payload)

    def write_file(self, path):
        with open(self._output_path(path), 'w') as f:
            f.writelines([
                self._format_header(),
                self._format_payload()
            ])

    @property
    def rows(self):
        return self._rows

    @property
    def columns(self):
        return self._columns

    @property
    def transform(self):
        return self._transform
