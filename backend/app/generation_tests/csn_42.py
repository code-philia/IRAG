try:
    from googleapiclient.errors import HttpError
except ImportError:
    class HttpError(Exception):
        """Fallback used only when the optional Google client is unavailable."""

        def __init__(self, resp=None, content=b""):
            super().__init__(content.decode("utf-8", errors="replace"))
            self.resp = resp
            self.content = content



class _DeleteRequest:
    def __init__(self, hook):
        self.hook = hook

    def execute(self, *, num_retries):
        self.hook.execute_retries = num_retries
        self.hook.events.append("execute")
        if self.hook.execute_error is not None:
            raise self.hook.execute_error
        return {"name": self.hook.operation_name}


class _Databases:
    def __init__(self, hook):
        self.hook = hook

    def delete(self, **kwargs):
        self.hook.delete_calls += 1
        self.hook.delete_arguments = kwargs
        return _DeleteRequest(self.hook)


class _Connection:
    def __init__(self, hook):
        self.hook = hook

    def databases(self):
        return _Databases(self.hook)


class _Hook:
    def __init__(self, operation_name, num_retries=3, execute_error=None):
        self.operation_name = operation_name
        self.num_retries = num_retries
        self.events = []
        self.delete_arguments = None
        self.execute_retries = None
        self.wait_arguments = None
        self.operation_complete = False
        self.delete_calls = 0
        self.execute_error = execute_error

    def get_conn(self):
        return _Connection(self)

    def _wait_for_operation_to_complete(self, *, project_id, operation_name):
        self.events.append("wait")
        self.wait_arguments = {
            "project_id": project_id,
            "operation_name": operation_name,
        }
        self.operation_complete = True


def test_deletes_the_requested_database_and_reuses_retry_configuration():
    hook = _Hook("operation-delete-analytics", num_retries=7)

    result = delete_database(hook, "prod-db", "analytics", "demo-project")

    assert result is None
    assert hook.delete_arguments == {
        "project": "demo-project",
        "instance": "prod-db",
        "database": "analytics",
    }
    assert hook.execute_retries == 7


def test_does_not_finish_when_request_acceptance_precedes_completion():
    hook = _Hook("operation-delete-analytics")

    delete_database(hook, "prod-db", "analytics", "demo-project")

    assert hook.events == ["execute", "wait"]
    assert hook.operation_complete is True
    assert hook.wait_arguments == {
        "project_id": "demo-project",
        "operation_name": "operation-delete-analytics",
    }


def test_waits_for_the_operation_returned_by_each_delete_request():
    hook = _Hook("operation-delete-audit")

    delete_database(hook, "staging-db", "audit", "staging-project")

    assert hook.wait_arguments == {
        "project_id": "staging-project",
        "operation_name": "operation-delete-audit",
    }


def test_forwards_default_project_id_consistently():
    hook = _Hook("operation-default", num_retries=5)

    delete_database(hook, "db-instance", "db-name")

    assert hook.delete_arguments == {
        "project": None,
        "instance": "db-instance",
        "database": "db-name",
    }
    assert hook.execute_retries == 5
    assert hook.wait_arguments == {
        "project_id": None,
        "operation_name": "operation-default",
    }


def test_does_not_wait_when_delete_request_fails():
    hook = _Hook(
        "operation-never-completes",
        execute_error=RuntimeError("delete request failed"),
    )

    try:
        delete_database(hook, "db-instance", "db-name", "project")
    except RuntimeError:
        pass
    else:
        raise AssertionError("delete_database should propagate the request failure")

    assert hook.events == ["execute"]
    assert hook.wait_arguments is None
    assert hook.operation_complete is False


def test_issues_exactly_one_delete_request():
    hook = _Hook("operation-once")

    delete_database(hook, "db-instance", "db-name", "project")

    assert hook.delete_calls == 1


def test_preserves_original_api_exception():
    class _FakeResponse:
        status = 500
        reason = "Internal Server Error"

    error = HttpError(
        resp=_FakeResponse(),
        content=b'{"error": "delete failed"}',
    )
    hook = _Hook(
        "operation-delete-failure",
        execute_error=error,
    )

    try:
        delete_database(hook, "db-instance", "db-name", "project")
    except HttpError as exc:
        assert exc is error
    else:
        raise AssertionError("delete_database should propagate the original HttpError")


TEST_CASES = [
    test_deletes_the_requested_database_and_reuses_retry_configuration,
    test_does_not_finish_when_request_acceptance_precedes_completion,
    test_waits_for_the_operation_returned_by_each_delete_request,
    test_forwards_default_project_id_consistently,
    test_does_not_wait_when_delete_request_fails,
    test_issues_exactly_one_delete_request,
    test_preserves_original_api_exception,
]
