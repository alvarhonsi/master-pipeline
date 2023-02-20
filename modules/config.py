import configparser

def read_config(filename):
    config = configparser.ConfigParser(converters={
        'tuple': parse_int_tuple, 
        "list": parse_int_list
    })
    config.read(filename)
    return config

def parse_int_tuple(input):
    return tuple(int(k.strip()) for k in input[1:-1].split(','))

def parse_int_list(input):
    if input == "[]":
        return []
    return [int(k.strip()) for k in input[1:-1].split(',')]