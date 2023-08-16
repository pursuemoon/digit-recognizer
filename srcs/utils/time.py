# -- coding: utf-8 --

def human_readable_time(seconds):
    hours = seconds // 3600
    minutes = seconds % 3600 // 60
    seconds = seconds % 60

    result  = ''
    result += f'{hours} hour' if hours > 0 else ''
    space = ' ' if len(result) > 0 else ''
    result += f'{space}{minutes} min' if minutes > 0 else ''
    space = ' ' if len(result) > 0 else ''
    result += f'{space}{seconds:.1f} sec' if seconds >= 0 else ''

    return result
