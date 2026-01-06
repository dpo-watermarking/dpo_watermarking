
import os

def load_file_by_line(path):

    with open(path, "r", encoding="utf-8") as f:

        return [line.strip() for line in f if len(line.strip()) > 0]
    
def path_wo_ext(path):

    return os.path.splitext(path)[0]

def break_text(texts):

    return [t.split() for t in texts]

def chunks(lst, n):

    for i in range(0, len(lst), n):

        yield lst[i:i + n]


