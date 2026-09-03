import astroid


def _function(source, function_name, occurrence=1):
    module = astroid.parse(source)
    functions = [
        node
        for node in module.nodes_of_class(astroid.FunctionDef)
        if node.name == function_name
    ]
    return functions[occurrence - 1]


def test_identifies_same_name_setter():
    node = _function(
        """
@property
def x(self):
    return self._x

@x.setter
def x(self, value):
    self._x = value
""",
        "x",
        occurrence=2,
    )

    assert redefined_by_decorator(node) is True


def test_identifies_same_name_deleter():
    node = _function(
        """
@property
def x(self):
    return self._x

@x.deleter
def x(self):
    del self._x
""",
        "x",
        occurrence=2,
    )

    assert redefined_by_decorator(node) is True


def test_rejects_different_name_setter():
    node = _function(
        """
@y.setter
def x(self, value):
    self._x = value
""",
        "x",
    )

    assert redefined_by_decorator(node) is False


def test_rejects_simple_decorator():
    node = _function(
        """
@staticmethod
def x():
    pass
""",
        "x",
    )

    assert redefined_by_decorator(node) is False


def test_rejects_method_without_decorator():
    node = _function(
        """
def x(self):
    return self._x
""",
        "x",
    )

    assert redefined_by_decorator(node) is False


def test_identifies_same_name_arbitrary_attribute_decorator():
    """The semantic rule is same-name dotted decoration, not setter-only."""
    node = _function(
        """
@x.cached
def x(self):
    return self._x
""",
        "x",
    )

    assert redefined_by_decorator(node) is True


def test_rejects_called_dotted_decorator():
    """A Call(Attribute(...)) is not the direct Attribute shape used by the GT rule."""
    node = _function(
        """
@x.setter()
def x(self, value):
    self._x = value
""",
        "x",
    )

    assert redefined_by_decorator(node) is False


def test_rejects_same_name_simple_decorator():
    """Matching text alone is insufficient; the decorator must be dotted."""
    node = _function(
        """
@x
def x(self):
    return self._x
""",
        "x",
    )

    assert redefined_by_decorator(node) is False


def test_identifies_matching_decorator_among_multiple():
    node = _function(
        """
@staticmethod
@x.deleter
def x(self):
    del self._x
""",
        "x",
    )

    assert redefined_by_decorator(node) is True


TEST_CASES = [
    test_identifies_same_name_setter,
    test_identifies_same_name_deleter,
    test_rejects_different_name_setter,
    test_rejects_simple_decorator,
    test_rejects_method_without_decorator,
    test_identifies_same_name_arbitrary_attribute_decorator,
    test_rejects_called_dotted_decorator,
    test_rejects_same_name_simple_decorator,
    test_identifies_matching_decorator_among_multiple,
]
