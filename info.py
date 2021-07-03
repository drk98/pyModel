import attrs
import datetime

VALID_SPECS = ["I", "t"]


@attrs(frozen=True, order=True)
class info:
    id: str = ""
    date: datetime.datetime = datetime.datetime.now()

    def __format__(self, format_spec):
        if format_spec[-1] not in VALID_SPECS:
            raise ValueError('{} format specifier not understood for this object', format_spec[:-1])

        if format_spec[-1] == "I":
            raw = str(self.id)

        elif format_spec[-1] == "t":
            raw = str(self.date)

        return "{r:{f}}".format(r=raw, f=format_spec)
