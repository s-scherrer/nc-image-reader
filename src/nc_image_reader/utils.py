import datetime


def mkdate(datestring):
    if len(datestring) == 10:
        return datetime.datetime.strptime(datestring, "%Y-%m-%d")
    elif len(datestring) == 16:
        return datetime.datetime.strptime(datestring, "%Y-%m-%dT%H:%M")
    else:  # pragma: no cover
        raise ValueError(f"Invalid date: {datestring}")
