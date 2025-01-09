from datetime import timedelta

OPEN_PARK_GATES = [201, 202, 203, 204, 205, 206, 207]
DOMESTIC_CAROUSELS = [0, 1, 2, 3, 4, 5, 6]

def total_seconds_to_hms(total_seconds: int, to_str: bool = False):
    """
    Convert total seconds to hours, minutes and seconds.
    total_seconds: int - The total number of seconds
    to_str: bool - If True, return the result as a string, otherwise as a tuple
    """
    hour = int(total_seconds // 3600)
    minute = int((total_seconds // 60) % 60)
    second = int(total_seconds % 60)

    if to_str:
        return f"{hour:02d}:{minute:02d}:{second:02d}"
    else:
        return hour, minute, second
    
def hms_to_total_seconds(hour: int, minute: int, second: int) -> int:
    """
    Convert hours, minutes and seconds to total seconds.
    hour: int - The number of hours
    minute: int - The number of minutes
    second: int - The number of seconds
    """
    return timedelta(hours=hour, minutes=minute, seconds=second).total_seconds()
    


