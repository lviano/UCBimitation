from os import path


def assets_dir(subfolder=None):
    if subfolder is None:
        return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../assets/'))
    else:
        return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../assets/'+subfolder))

def deterministic_assets_dir(subfolder=None):
    if subfolder is None:
        return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../deterministic_assets/'))
    else:
        return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../deterministic_assets/'+subfolder))
