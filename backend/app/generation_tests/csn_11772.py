from types import SimpleNamespace


def test_returns_extension_for_matching_compiler_mimetype():
    asset = SimpleNamespace(
        environment=SimpleNamespace(mimetypes={".js": "application/javascript", ".coffee": "text/coffeescript"}),
        compiler_mimetype="text/coffeescript",
    )
    assert compiler_format_extension(asset) == ".coffee"


def test_returns_none_when_no_registered_mimetype_matches():
    asset = SimpleNamespace(
        environment=SimpleNamespace(mimetypes={".js": "application/javascript"}),
        compiler_mimetype="text/coffeescript",
    )
    assert compiler_format_extension(asset) is None


def test_uses_the_extension_owned_by_the_matching_registry_entry():
    asset = SimpleNamespace(
        environment=SimpleNamespace(mimetypes={".html": "text/html", ".tmpl": "text/x-template", ".txt": "text/plain"}),
        compiler_mimetype="text/x-template",
    )
    assert compiler_format_extension(asset) == ".tmpl"


TEST_CASES = [
    test_returns_extension_for_matching_compiler_mimetype,
    test_returns_none_when_no_registered_mimetype_matches,
    test_uses_the_extension_owned_by_the_matching_registry_entry,
]
