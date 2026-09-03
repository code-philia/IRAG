class _DeleteRequest:
    def __init__(self, hook):
        self.hook = hook

    def execute(self, *, num_retries):
        self.hook.execute_retries = num_retries
        self.hook.events.append("execute")
        return {"name": self.hook.operation_name}


class _Databases:
    def __init__(self, hook):
        self.hook = hook

    def delete(self, **kwargs):
        self.hook.delete_arguments = kwargs
        return _DeleteRequest(self.hook)


class _Connection:
    def __init__(self, hook):
        self.hook = hook

    def databases(self):
        return _Databases(self.hook)


class _Hook:
    def __init__(self, operation_name, num_retries=3):
        self.operation_name = operation_name
        self.num_retries = num_retries
        self.events = []
        self.delete_arguments = None
        self.execute_retries = None
        self.wait_arguments = None
        self.operation_complete = False

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


TEST_CASES = [
    test_deletes_the_requested_database_and_reuses_retry_configuration,
    test_does_not_finish_when_request_acceptance_precedes_completion,
    test_waits_for_the_operation_returned_by_each_delete_request,
]
