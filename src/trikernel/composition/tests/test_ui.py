from trikernel.composition.ui import TerminalUI


def test_write_output(capsys):
    ui = TerminalUI()
    ui.write_output("hello")
    captured = capsys.readouterr()
    assert captured.out == "hello\n"


def test_write_stream(capsys):
    ui = TerminalUI()
    ui.write_stream(["a", "b", "c"], end="")
    captured = capsys.readouterr()
    assert captured.out == "abc"
