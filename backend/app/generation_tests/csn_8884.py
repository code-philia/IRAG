import ast


def _filter_dead_code(statements):
    kept = []
    for statement in statements:
        kept.append(statement)
        if isinstance(statement, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
            break
    return kept


def _try_node(source):
    tree = ast.parse(source)
    return next(node for node in ast.walk(tree) if isinstance(node, ast.Try))


def _has_call(nodes, function_name):
    return any(
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == function_name
        for statement in nodes
    )


class _OptimizerContext:
    def generic_visit(self, node):
        return node


def test_removes_dead_code_from_try_body():
    node = _try_node(
        """
def sample():
    try:
        return_value = 1
        return return_value
        dead = 2
    except Exception:
        handle()
"""
    )

    result = visit_Try(_OptimizerContext(), node)

    assert isinstance(result, ast.Try)
    assert len(result.body) == 2 and isinstance(result.body[-1], ast.Return)
    assert len(result.handlers) == 1


def test_removes_dead_code_from_all_try_statement_regions():
    node = _try_node(
        """
def sample():
    try:
        work()
        return x
        dead_in_try()
    except Exception:
        recover()
    else:
        return y
        dead_in_else()
    finally:
        raise RuntimeError()
        dead_in_finally()
"""
    )

    result = visit_Try(_OptimizerContext(), node)

    assert isinstance(result, ast.Try)
    assert isinstance(result.body[-1], ast.Return)
    assert not _has_call(result.body, "dead_in_try")
    assert isinstance(result.orelse[-1], ast.Return)
    assert not _has_call(result.orelse, "dead_in_else")
    assert isinstance(result.finalbody[-1], ast.Raise)
    assert not _has_call(result.finalbody, "dead_in_finally")
    assert len(result.handlers) == 1


def test_preserves_try_source_location():
    node = _try_node(
        """
def sample():
    try:
        return x
        dead()
    except Exception:
        handle()
"""
    )

    result = visit_Try(_OptimizerContext(), node)

    assert isinstance(result, ast.Try)
    assert (result.lineno, result.col_offset) == (node.lineno, node.col_offset)
    if hasattr(node, "end_lineno"):
        assert (result.end_lineno, result.end_col_offset) == (node.end_lineno, node.end_col_offset)


def test_preserves_try_handlers_and_reachable_structure():
    node = _try_node(
        """
def sample():
    try:
        work()
    except ValueError:
        recover_value()
    except TypeError:
        recover_type()
    else:
        finish()
    finally:
        cleanup()
"""
    )

    result = visit_Try(_OptimizerContext(), node)

    assert isinstance(result, ast.Try)
    assert len(result.handlers) == 2
    assert isinstance(result.handlers[0].type, ast.Name)
    assert result.handlers[0].type.id == "ValueError"
    assert isinstance(result.handlers[1].type, ast.Name)
    assert result.handlers[1].type.id == "TypeError"
    assert len(result.body) == 1
    assert len(result.orelse) == 1
    assert len(result.finalbody) == 1


def test_removes_dead_code_after_break_in_try_body():
    node = _try_node(
        """
def sample():
    while True:
        try:
            work()
            break
            dead()
        except Exception:
            recover()
"""
    )

    result = visit_Try(_OptimizerContext(), node)

    assert isinstance(result.body[-1], ast.Break)
    assert not _has_call(result.body, "dead")


def test_removes_dead_code_after_continue_in_try_body():
    node = _try_node(
        """
def sample():
    while True:
        try:
            work()
            continue
            dead()
        except Exception:
            recover()
"""
    )

    result = visit_Try(_OptimizerContext(), node)

    assert isinstance(result.body[-1], ast.Continue)
    assert not _has_call(result.body, "dead")


def test_preserves_reachable_handler_body():
    node = _try_node(
        """
def sample():
    try:
        work()
    except ValueError:
        first()
        second()
"""
    )

    result = visit_Try(_OptimizerContext(), node)

    calls = [
        statement.value.func.id
        for statement in result.handlers[0].body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
    ]

    assert calls == ["first", "second"]


def test_rebuilds_try_without_mutating_original_body():
    """Repository-structural contract: visit_Try returns a rebuilt Try node."""
    node = _try_node(
        """
def sample():
    try:
        work()
        return 1
        dead()
    except Exception:
        recover()
"""
    )

    original_body = list(node.body)
    result = visit_Try(_OptimizerContext(), node)

    assert result is not node
    assert node.body == original_body
    assert len(result.body) == 2
    assert isinstance(result.body[-1], ast.Return)


TEST_CASES = [
    test_removes_dead_code_from_try_body,
    test_removes_dead_code_from_all_try_statement_regions,
    test_preserves_try_source_location,
    test_preserves_try_handlers_and_reachable_structure,
    test_removes_dead_code_after_break_in_try_body,
    test_removes_dead_code_after_continue_in_try_body,
    test_preserves_reachable_handler_body,
    test_rebuilds_try_without_mutating_original_body,
]
