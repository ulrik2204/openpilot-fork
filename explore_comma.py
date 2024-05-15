import json
import re
from calendar import c
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# This is taken from the Openpilot project
from openpilot.tools.lib.logreader import LogReader


def read_video_frames(file_path: str):
    frames = []
    video = cv2.VideoCapture(file_path)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()
    return frames


def isiterable(item):
    try:
        iter(item)
        return True
    except TypeError:
        return False


def to_json(o, print_progress=True, depth=0, chain=""):
    if depth > 3:
        return str(o)

    if (
        isinstance(o, str)
        or isinstance(o, int)
        or isinstance(o, float)
        or isinstance(o, bool)
        or isinstance(o, bytes)
        or o is None
    ):
        return str(o)
    elif (
        isinstance(o, list)
        or isinstance(o, tuple)
        or isinstance(o, set)
        or isiterable(o)
    ):
        values = []
        for item in o:
            field = to_json(item, print_progress, depth=depth + 1)
            values.append(field)
        return values
    elif isinstance(o, dict):
        total_dict = dict()
        for key, value in o.items():
            total_dict[key] = to_json(value, print_progress, depth=depth + 1)
        return total_dict
    # If it is a non-primitive builtin class
    elif hasattr(o.__class__, "__module__") and not o.__class__.__module__.startswith(
        "builtins"
    ):
        item_attrs = [attr for attr in dir(o) if not attr.startswith("_")]
        # print("item_attrs", item_attrs)
        total_dict = dict()
        progress = 0
        for attr in item_attrs:
            if print_progress:
                progress += 1
                print(
                    f"Progress {depth}: {progress}/{len(item_attrs)} chain={chain}",
                )
            try:
                total_dict[attr] = to_json(
                    getattr(o, attr),
                    print_progress,
                    depth=depth + 1,
                    chain=chain + f".{attr}",
                )
            except Exception:
                continue
        return total_dict
    return str(o)


def parse_capnp_string(s: str):
    # Wrap in quotes and replace = with :
    s = re.sub(r"(\w+) =", r'"\1": ', s)

    # Replace ( with [ for lists
    pattern = r'\((?:"[^"]*"|\d+)(?:\s*,\s*(?:"[^"]*"|\d+))*\)'

    def convert(match):
        inner_text = match.group(0)[1:-1]  # Remove the outer parentheses
        return "[" + inner_text + "]"  # Surround with square brackets

    s = re.sub(pattern, convert, s)
    # Replace the rest of the parentheses with curly braces and remove newlines
    s.replace("( ", "{ ").replace(" )", " }").replace("\n", "")
    return s

    #


def main():
    device_path = Path("/home/ulrikro/datasets/CommaAI/2024-01-14--13-01-26--10/")
    ecamera_path = device_path / "ecamera.hevc"
    fcamera_path = device_path / "fcamera.hevc"
    qcamera_path = device_path / "qcamera.ts"
    qlog_path = device_path / "qlog"
    rlog_path = device_path / "rlog"

    qlog_data = LogReader(qlog_path.as_posix())
    qlog_list = list(qlog_data)
    index = 1
    string = to_json(qlog_list[index])
    with open(f"qlog{index}.json", "w") as f:
        json.dump(string, f, indent=4)


if __name__ == "__main__":
    main()
