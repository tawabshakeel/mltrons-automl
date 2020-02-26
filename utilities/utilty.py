from libraries import *


class Utility(object):
    def __init__(self):
        pass

    @classmethod
    def random_string_generate(cls, len_of_str=7):
        return ''.join(choice(ascii_uppercase) for i in range(len_of_str))

    @staticmethod
    def create_directory(path_base,directory_name):
        path_base = '{}/{}'.format(path_base, directory_name)
        if not os.path.exists(path_base):
            os.makedirs(path_base)
        return path_base