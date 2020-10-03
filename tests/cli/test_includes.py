import os

import xmltodict

from microstructpy import cli


def test_equivalence():
    print(os.getcwd())
    files_dir = os.path.join(os.path.dirname(__file__), 'test_includes_files')
    with open(os.path.join(files_dir, 'expected_input.xml'), 'r') as file:
        expected = xmltodict.parse(file.read())

    for fname in ['input.xml', 'different_dir/input.xml']:
        actual = cli.input2dict(os.path.join(files_dir, fname))
        assert expected == actual
