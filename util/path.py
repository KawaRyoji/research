import os


def dir2paths(dir_path):
    paths = list(
            map(
                lambda x: os.path.join(dir_path, x),
                os.listdir(dir_path),
            )
        )
    
    return paths