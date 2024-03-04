def clean_line(line: str) -> str:
    """
    Clear the line from new line characters and comments
    :param line:
    :return: str
    """
    new_line = line.strip()

    if '#' in new_line:
        return None
    if new_line == '':
        return None
    else:
        return new_line


def get_requirements(filename: str) -> list:
    """
    Reads a requirement file and returns the list of requirements.

    :param filename: str
    :return: list of requirements
    """
    output = []

    with open(filename, 'r') as fp:
        lines: list = fp.readlines()

    for line in lines:
        new_line = clean_line(line)
        if new_line:
            output.append(new_line)

    return output
