import numpy as np
from typing import List

"""
Wrap the representation of a HIC image to make it easier to write/retrieve from byte stream
"""


class HicImage:
    def __init__(self, rows: int, columns: int, style: str, transform_params: List[str], payload: np.ndarray):
        self._rows = rows
        self._columns = columns
        self._style = style
        self._params = transform_params
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
            header_content["style"],
            header_content["transform_params"],
            content
        )

    @staticmethod
    def _parse_header(header):
        header = header.replace("\n", "")
        splits = header.split(" ")
        return {
            "rows": int(splits[0]),
            "columns": int(splits[1]),
            "style": splits[2],
            "transform_params": splits[3:]
        }

    def _output_path(self, path: str):
        if path.endswith(".hic"):
            return path
        return path + ".hic"

    def _format_header(self):
        return "{} {} {} {}".format(self._rows, self._columns, self._style,
                                    " ".join(self._params))

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
    def style(self):
        return self._style
