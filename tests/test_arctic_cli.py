from applet.cli import parse_args


def test_arctic_flag_is_available(monkeypatch):
    monkeypatch.setattr("sys.argv", ["applet", "run", "-i", "input", "-o", "output", "--arctic"])
    args = parse_args()
    assert args.arctic is True