#--------------------------------------
# Simple utilities
#--------------------------------------
from datetime import datetime

def url_join(path1, path2, trailing_slash=False):
    url = path1.rstrip('/') + '/' + path2.lstrip('/')
    url += '/' if trailing_slash else ''
    return url

def datetime_to_tuple(dt):
    return (dt.year, dt.month, dt.day,
            dt.hour, dt.minute, dt.second, dt.microsecond)

def tuple_to_datetime(dt_list):
    if isinstance(dt_list, (list, tuple)):
        return datetime(*dt_list)
    else:
        return datetime.today()
