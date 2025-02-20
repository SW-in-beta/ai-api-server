from enum import Enum

class CaseInsensitiveStrEnum(str, Enum):
    """
    대소문자 구분 없이 enum 멤버를 찾기 위해 _missing_을 오버라이드합니다.
    """
    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            uppered = value.upper()
            if uppered in cls:
                return cls[uppered]
        return super()._missing_(value)

class BinaryOperation(CaseInsensitiveStrEnum, Enum):
  AND = 'AND'
  OR = 'OR'
  NAND = 'NAND'
  NOR = 'NOR'
  XOR = 'XOR'
  
class UnaryOperation(CaseInsensitiveStrEnum, Enum):
  NOT = 'NOT'