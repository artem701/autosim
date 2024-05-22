
from enum import Enum

class SmartEnum(Enum):
    @classmethod
    def from_string(cls, string: str):
        for item in cls:
            if item.name == string:
                return item
        raise RuntimeError(f"item {string} not in enum {cls.name}")
