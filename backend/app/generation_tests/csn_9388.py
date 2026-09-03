from urllib import parse as urlparse


def test_removes_query_and_preserves_path_and_fragment():
    result = url_dequery("https://example.com/report.csv?download=1&user=7#summary")

    assert result == "https://example.com/report.csv#summary"


def test_removes_query_without_changing_scheme_host_or_path():
    result = url_dequery("http://api.example.test:8080/v1/items;latest?limit=10")

    assert result == "http://api.example.test:8080/v1/items;latest"


def test_leaves_url_without_query_unchanged():
    result = url_dequery("https://example.com/landing#top")

    assert result == "https://example.com/landing#top"


def test_preserves_userinfo_port_params_encoding_and_fragment():
    result = url_dequery(
        "https://user:pw@example.com:8443/a%20b;v=1?x=1#frag"
    )

    assert result == "https://user:pw@example.com:8443/a%20b;v=1#frag"


def test_handles_relative_url_and_preserves_fragment():
    result = url_dequery("docs/page?mode=print#section")

    assert result == "docs/page#section"


def test_removes_empty_query_delimiter():
    result = url_dequery("https://example.com/path?#frag")

    assert result == "https://example.com/path#frag"


def test_handles_scheme_relative_url():
    result = url_dequery("//example.com/path?x=1#f")

    assert result == "//example.com/path#f"


TEST_CASES = [
    test_removes_query_and_preserves_path_and_fragment,
    test_removes_query_without_changing_scheme_host_or_path,
    test_leaves_url_without_query_unchanged,
    test_preserves_userinfo_port_params_encoding_and_fragment,
    test_handles_relative_url_and_preserves_fragment,
    test_removes_empty_query_delimiter,
    test_handles_scheme_relative_url,
]
