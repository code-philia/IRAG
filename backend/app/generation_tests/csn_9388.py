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


TEST_CASES = [
    test_removes_query_and_preserves_path_and_fragment,
    test_removes_query_without_changing_scheme_host_or_path,
    test_leaves_url_without_query_unchanged,
]
