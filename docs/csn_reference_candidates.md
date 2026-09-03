# Study Dev Reference Candidates

> Snapshot of the 20 participant-facing candidates for the four active cases.

> Generated from the study-dev candidate API; `csn_11772` is intentionally omitted.

## csn_8884 - Try AST dead-code cleanup

Candidates exported: 20

### Rank 1 - `code_1012324` - `MisdesignChecker.visit_tryexcept`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/design_analysis.py`

- Similarity: `0.646205`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/design_analysis.py#L473-L479


```python

def visit_tryexcept(self, node):
        """increments the branches counter"""
        branches = len(node.handlers)
        if node.orelse:
            branches += 1
        self._inc_branch(node, branches)
        self._inc_all_stmts(branches)

```

### Rank 2 - `code_1028080` - `Parser.except_clause`

- Repository: `google/grumpy`

- Path: `third_party/pythonparser/parser.py`

- Similarity: `0.616341`

- Source URL: https://github.com/google/grumpy/blob/3ec87959189cfcdeae82eb68a47648ac25ceb10b/third_party/pythonparser/parser.py#L1324-L1345


```python

def except_clause(self, except_loc, exc_opt):
        """
        (2.6, 2.7) except_clause: 'except' [test [('as' | ',') test]]
        (3.0-) except_clause: 'except' [test ['as' NAME]]
        """
        type_ = name = as_loc = name_loc = None
        loc = except_loc
        if exc_opt:
            type_, name_opt = exc_opt
            loc = loc.join(type_.loc)
            if name_opt:
                as_loc, name_tok, name_node = name_opt
                if name_tok:
                    name = name_tok.value
                    name_loc = name_tok.loc
                else:
                    name = name_node
                    name_loc = name_node.loc
                loc = loc.join(name_loc)
        return ast.ExceptHandler(type=type_, name=name,
                                 except_loc=except_loc, as_loc=as_loc, name_loc=name_loc,
                                 loc=loc)

```

### Rank 3 - `code_1037534` - `ExceptionsChecker.visit_tryexcept`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/exceptions.py`

- Similarity: `0.611775`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/exceptions.py#L499-L568


```python

def visit_tryexcept(self, node):
        """check for empty except"""
        self._check_try_except_raise(node)
        exceptions_classes = []
        nb_handlers = len(node.handlers)
        for index, handler in enumerate(node.handlers):
            if handler.type is None:
                if not _is_raising(handler.body):
                    self.add_message("bare-except", node=handler)

                # check if an "except:" is followed by some other
                # except
                if index < (nb_handlers - 1):
                    msg = "empty except clause should always appear last"
                    self.add_message("bad-except-order", node=node, args=msg)

            elif isinstance(handler.type, astroid.BoolOp):
                self.add_message(
                    "binary-op-exception", node=handler, args=handler.type.op
                )
            else:
                try:
                    excs = list(_annotated_unpack_infer(handler.type))
                except astroid.InferenceError:
                    continue

                for part, exc in excs:
                    if exc is astroid.Uninferable:
                        continue
                    if isinstance(exc, astroid.Instance) and utils.inherit_from_std_ex(
                        exc
                    ):
                        # pylint: disable=protected-access
                        exc = exc._proxied

                    self._check_catching_non_exception(handler, exc, part)

                    if not isinstance(exc, astroid.ClassDef):
                        continue

                    exc_ancestors = [
                        anc
                        for anc in exc.ancestors()
                        if isinstance(anc, astroid.ClassDef)
                    ]

                    for previous_exc in exceptions_classes:
                        if previous_exc in exc_ancestors:
                            msg = "%s is an ancestor class of %s" % (
                                previous_exc.name,
                                exc.name,
                            )
                            self.add_message(
                                "bad-except-order", node=handler.type, args=msg
                            )
                    if (
                        exc.name in self.config.overgeneral_exceptions
                        and exc.root().name == utils.EXCEPTIONS_MODULE
                        and not _is_raising(handler.body)
                    ):
                        self.add_message(
                            "broad-except", args=exc.name, node=handler.type
                        )

                    if exc in exceptions_classes:
                        self.add_message(
                            "duplicate-except", args=exc.name, node=handler.type
                        )

                exceptions_classes += [exc for _, exc in excs]

```

### Rank 4 - `code_1039906` - `is_node_inside_try_except`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/utils.py`

- Similarity: `0.602592`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/utils.py#L877-L888


```python

def is_node_inside_try_except(node: astroid.Raise) -> bool:
    """Check if the node is directly under a Try/Except statement.
    (but not under an ExceptHandler!)

    Args:
        node (astroid.Raise): the node raising the exception.

    Returns:
        bool: True if the node is inside a try/except statement, False otherwise.
    """
    context = find_try_except_wrapper_node(node)
    return isinstance(context, astroid.TryExcept)

```

### Rank 5 - `code_1043640` - `PythonASTOptimizer.visit_ExceptHandler`

- Repository: `chrisrink10/basilisp`

- Path: `src/basilisp/lang/compiler/optimizer.py`

- Similarity: `0.587081`

- Source URL: https://github.com/chrisrink10/basilisp/blob/3d82670ee218ec64eb066289c82766d14d18cc92/src/basilisp/lang/compiler/optimizer.py#L18-L29


```python

def visit_ExceptHandler(self, node: ast.ExceptHandler) -> Optional[ast.AST]:
        """Eliminate dead code from except handler bodies."""
        new_node = self.generic_visit(node)
        assert isinstance(new_node, ast.ExceptHandler)
        return ast.copy_location(
            ast.ExceptHandler(
                type=new_node.type,
                name=new_node.name,
                body=_filter_dead_code(new_node.body),
            ),
            new_node,
        )

```

### Rank 6 - `code_1004901` - `OverlappingExceptionsChecker.visit_tryexcept`

- Repository: `PyCQA/pylint`

- Path: `pylint/extensions/overlapping_exceptions.py`

- Similarity: `0.563230`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/extensions/overlapping_exceptions.py#L34-L83


```python

def visit_tryexcept(self, node):
        """check for empty except"""
        for handler in node.handlers:
            if handler.type is None:
                continue
            if isinstance(handler.type, astroid.BoolOp):
                continue
            try:
                excs = list(_annotated_unpack_infer(handler.type))
            except astroid.InferenceError:
                continue

            handled_in_clause = []
            for part, exc in excs:
                if exc is astroid.Uninferable:
                    continue
                if isinstance(exc, astroid.Instance) and utils.inherit_from_std_ex(exc):
                    # pylint: disable=protected-access
                    exc = exc._proxied

                if not isinstance(exc, astroid.ClassDef):
                    continue

                exc_ancestors = [
                    anc for anc in exc.ancestors() if isinstance(anc, astroid.ClassDef)
                ]

                for prev_part, prev_exc in handled_in_clause:
                    prev_exc_ancestors = [
                        anc
                        for anc in prev_exc.ancestors()
                        if isinstance(anc, astroid.ClassDef)
                    ]
                    if exc == prev_exc:
                        self.add_message(
                            "overlapping-except",
                            node=handler.type,
                            args="%s and %s are the same"
                            % (prev_part.as_string(), part.as_string()),
                        )
                    elif prev_exc in exc_ancestors or exc in prev_exc_ancestors:
                        ancestor = part if exc in prev_exc_ancestors else prev_part
                        descendant = part if prev_exc in exc_ancestors else prev_part
                        self.add_message(
                            "overlapping-except",
                            node=handler.type,
                            args="%s is an ancestor class of %s"
                            % (ancestor.as_string(), descendant.as_string()),
                        )
                handled_in_clause += [(part, exc)]

```

### Rank 7 - `code_1043461` - `_filter_dead_code`

- Repository: `chrisrink10/basilisp`

- Path: `src/basilisp/lang/compiler/optimizer.py`

- Similarity: `0.559611`

- Source URL: https://github.com/chrisrink10/basilisp/blob/3d82670ee218ec64eb066289c82766d14d18cc92/src/basilisp/lang/compiler/optimizer.py#L5-L14


```python

def _filter_dead_code(nodes: Iterable[ast.AST]) -> List[ast.AST]:
    """Return a list of body nodes, trimming out unreachable code (any
    statements appearing after `break`, `continue`, and `return` nodes)."""
    new_nodes: List[ast.AST] = []
    for node in nodes:
        if isinstance(node, (ast.Break, ast.Continue, ast.Return)):
            new_nodes.append(node)
            break
        new_nodes.append(node)
    return new_nodes

```

### Rank 8 - `code_1012343` - `get_exception_handlers`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/utils.py`

- Similarity: `0.539866`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/utils.py#L856-L874


```python

def get_exception_handlers(
    node: astroid.node_classes.NodeNG, exception=Exception
) -> List[astroid.ExceptHandler]:
    """Return the collections of handlers handling the exception in arguments.

    Args:
        node (astroid.NodeNG): A node that is potentially wrapped in a try except.
        exception (builtin.Exception or str): exception or name of the exception.

    Returns:
        list: the collection of handlers that are handling the exception or None.

    """
    context = find_try_except_wrapper_node(node)
    if isinstance(context, astroid.TryExcept):
        return [
            handler for handler in context.handlers if error_of_type(handler, exception)
        ]
    return None

```

### Rank 9 - `code_1010686` - `Python3Checker.visit_excepthandler`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/python3.py`

- Similarity: `0.532151`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/python3.py#L1271-L1316


```python

def visit_excepthandler(self, node):
        """Visit an except handler block and check for exception unpacking."""

        def _is_used_in_except_block(node):
            scope = node.scope()
            current = node
            while (
                current
                and current != scope
                and not isinstance(current, astroid.ExceptHandler)
            ):
                current = current.parent
            return isinstance(current, astroid.ExceptHandler) and current.type != node

        if isinstance(node.name, (astroid.Tuple, astroid.List)):
            self.add_message("unpacking-in-except", node=node)
            return

        if not node.name:
            return

        # Find any names
        scope = node.parent.scope()
        scope_names = scope.nodes_of_class(astroid.Name, skip_klass=astroid.FunctionDef)
        scope_names = list(scope_names)
        potential_leaked_names = [
            scope_name
            for scope_name in scope_names
            if scope_name.name == node.name.name
            and scope_name.lineno > node.lineno
            and not _is_used_in_except_block(scope_name)
        ]
        reassignments_for_same_name = {
            assign_name.lineno
            for assign_name in scope.nodes_of_class(
                astroid.AssignName, skip_klass=astroid.FunctionDef
            )
            if assign_name.name == node.name.name
        }
        for leaked_name in potential_leaked_names:
            if any(
                node.lineno < elem < leaked_name.lineno
                for elem in reassignments_for_same_name
            ):
                continue
            self.add_message("exception-escape", node=leaked_name)

```

### Rank 10 - `code_1014538` - `LoggingVisitor.visit_ExceptHandler`

- Repository: `globality-corp/flake8-logging-format`

- Path: `logging_format/visitor.py`

- Similarity: `0.526310`

- Source URL: https://github.com/globality-corp/flake8-logging-format/blob/3c6ce53d0ff1ec369799cff0ed6d048343252e40/logging_format/visitor.py#L170-L182


```python

def visit_ExceptHandler(self, node):
        """
        Process except blocks.

        """
        name = self.get_except_handler_name(node)
        if not name:
            super(LoggingVisitor, self).generic_visit(node)
            return

        self.current_except_names.append(name)
        super(LoggingVisitor, self).generic_visit(node)
        self.current_except_names.pop()

```

### Rank 11 - `code_1002254` - `is_inside_except`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/utils.py`

- Similarity: `0.514321`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/utils.py#L209-L215


```python

def is_inside_except(node):
    """Returns true if node is inside the name of an except handler."""
    current = node
    while current and not isinstance(current.parent, astroid.ExceptHandler):
        current = current.parent

    return current and current is current.parent.name

```

### Rank 12 - `code_1032602` - `find_try_except_wrapper_node`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/utils.py`

- Similarity: `0.506358`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/utils.py#L812-L823


```python

def find_try_except_wrapper_node(
    node: astroid.node_classes.NodeNG
) -> Union[astroid.ExceptHandler, astroid.TryExcept]:
    """Return the ExceptHandler or the TryExcept node in which the node is."""
    current = node
    ignores = (astroid.ExceptHandler, astroid.TryExcept)
    while current and not isinstance(current.parent, ignores):
        current = current.parent

    if current and isinstance(current.parent, ignores):
        return current.parent
    return None

```

### Rank 13 - `code_1041598` - `LoggingVisitor.is_bare_exception`

- Repository: `globality-corp/flake8-logging-format`

- Path: `logging_format/visitor.py`

- Similarity: `0.504919`

- Source URL: https://github.com/globality-corp/flake8-logging-format/blob/3c6ce53d0ff1ec369799cff0ed6d048343252e40/logging_format/visitor.py#L251-L256


```python

def is_bare_exception(self, node):
        """
        Checks if the node is a bare exception name from an except block.

        """
        return isinstance(node, Name) and node.id in self.current_except_names

```

### Rank 14 - `code_1023813` - `exons`

- Repository: `Clinical-Genomics/scout`

- Path: `scout/commands/delete/delete_command.py`

- Similarity: `0.504262`

- Source URL: https://github.com/Clinical-Genomics/scout/blob/90a551e2e1653a319e654c2405c2866f93d0ebb9/scout/commands/delete/delete_command.py#L93-L98


```python

def exons(context, build):
    """Delete all exons in the database"""
    LOG.info("Running scout delete exons")
    adapter = context.obj['adapter']

    adapter.drop_exons(build)

```

### Rank 15 - `code_1015880` - `BasicChecker._check_not_in_finally`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/base.py`

- Similarity: `0.504213`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/base.py#L1379-L1395


```python

def _check_not_in_finally(self, node, node_name, breaker_classes=()):
        """check that a node is not inside a finally clause of a
        try...finally statement.
        If we found before a try...finally bloc a parent which its type is
        in breaker_classes, we skip the whole check."""
        # if self._tryfinallys is empty, we're not an in try...finally block
        if not self._tryfinallys:
            return
        # the node could be a grand-grand...-children of the try...finally
        _parent = node.parent
        _node = node
        while _parent and not isinstance(_parent, breaker_classes):
            if hasattr(_parent, "finalbody") and _node in _parent.finalbody:
                self.add_message("lost-exception", node=node, args=node_name)
                return
            _node = _parent
            _parent = _node.parent

```

### Rank 16 - `code_1015672` - `node_ignores_exception`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/utils.py`

- Similarity: `0.499591`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/utils.py#L891-L902


```python

def node_ignores_exception(
    node: astroid.node_classes.NodeNG, exception=Exception
) -> bool:
    """Check if the node is in a TryExcept which handles the given exception.

    If the exception is not given, the function is going to look for bare
    excepts.
    """
    managing_handlers = get_exception_handlers(node, exception)
    if not managing_handlers:
        return False
    return any(managing_handlers)

```

### Rank 17 - `code_1036608` - `is_from_fallback_block`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/utils.py`

- Similarity: `0.489709`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/utils.py#L826-L846


```python

def is_from_fallback_block(node: astroid.node_classes.NodeNG) -> bool:
    """Check if the given node is from a fallback import block."""
    context = find_try_except_wrapper_node(node)
    if not context:
        return False

    if isinstance(context, astroid.ExceptHandler):
        other_body = context.parent.body
        handlers = context.parent.handlers
    else:
        other_body = itertools.chain.from_iterable(
            handler.body for handler in context.handlers
        )
        handlers = context.handlers

    has_fallback_imports = any(
        isinstance(import_node, (astroid.ImportFrom, astroid.Import))
        for import_node in other_body
    )
    ignores_import_error = _except_handlers_ignores_exception(handlers, ImportError)
    return ignores_import_error or has_fallback_imports

```

### Rank 18 - `code_1043790` - `clobber_in_except`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/utils.py`

- Similarity: `0.489417`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/utils.py#L240-L262


```python

def clobber_in_except(
    node: astroid.node_classes.NodeNG
) -> Tuple[bool, Tuple[str, str]]:
    """Checks if an assignment node in an except handler clobbers an existing
    variable.

    Returns (True, args for W0623) if assignment clobbers an existing variable,
    (False, None) otherwise.
    """
    if isinstance(node, astroid.AssignAttr):
        return True, (node.attrname, "object %r" % (node.expr.as_string(),))
    if isinstance(node, astroid.AssignName):
        name = node.name
        if is_builtin(name):
            return (True, (name, "builtins"))

        stmts = node.lookup(name)[1]
        if stmts and not isinstance(
            stmts[0].assign_type(),
            (astroid.Assign, astroid.AugAssign, astroid.ExceptHandler),
        ):
            return True, (name, "outer scope (line %s)" % stmts[0].fromlineno)
    return False, None

```

### Rank 19 - `code_1037136` - `PythonASTOptimizer.visit_While`

- Repository: `chrisrink10/basilisp`

- Path: `src/basilisp/lang/compiler/optimizer.py`

- Similarity: `0.469420`

- Source URL: https://github.com/chrisrink10/basilisp/blob/3d82670ee218ec64eb066289c82766d14d18cc92/src/basilisp/lang/compiler/optimizer.py#L75-L86


```python

def visit_While(self, node: ast.While) -> Optional[ast.AST]:
        """Eliminate dead code from while bodies."""
        new_node = self.generic_visit(node)
        assert isinstance(new_node, ast.While)
        return ast.copy_location(
            ast.While(
                test=new_node.test,
                body=_filter_dead_code(new_node.body),
                orelse=_filter_dead_code(new_node.orelse),
            ),
            new_node,
        )

```

### Rank 20 - `code_1002345` - `PythonASTOptimizer.visit_If`

- Repository: `chrisrink10/basilisp`

- Path: `src/basilisp/lang/compiler/optimizer.py`

- Similarity: `0.463583`

- Source URL: https://github.com/chrisrink10/basilisp/blob/3d82670ee218ec64eb066289c82766d14d18cc92/src/basilisp/lang/compiler/optimizer.py#L62-L73


```python

def visit_If(self, node: ast.If) -> Optional[ast.AST]:
        """Eliminate dead code from if/elif bodies."""
        new_node = self.generic_visit(node)
        assert isinstance(new_node, ast.If)
        return ast.copy_location(
            ast.If(
                test=new_node.test,
                body=_filter_dead_code(new_node.body),
                orelse=_filter_dead_code(new_node.orelse),
            ),
            new_node,
        )

```

## csn_3846 - Decorator redefinition

Candidates exported: 20

### Rank 1 - `code_1023534` - `decorated_with`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/utils.py`

- Similarity: `0.642000`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/utils.py#L738-L749


```python

def decorated_with(func: astroid.FunctionDef, qnames: Iterable[str]) -> bool:
    """Determine if the `func` node has a decorator with the qualified name `qname`."""
    decorators = func.decorators.nodes if func.decorators else []
    for decorator_node in decorators:
        try:
            if any(
                i is not None and i.qname() in qnames for i in decorator_node.infer()
            ):
                return True
        except astroid.InferenceError:
            continue
    return False

```

### Rank 2 - `code_1002786` - `ClassChecker.visit_functiondef`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/classes.py`

- Similarity: `0.606322`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/classes.py#L855-L934


```python

def visit_functiondef(self, node):
        """check method arguments, overriding"""
        # ignore actual functions
        if not node.is_method():
            return

        self._check_useless_super_delegation(node)

        klass = node.parent.frame()
        self._meth_could_be_func = True
        # check first argument is self if this is actually a method
        self._check_first_arg_for_type(node, klass.type == "metaclass")
        if node.name == "__init__":
            self._check_init(node)
            return
        # check signature if the method overloads inherited method
        for overridden in klass.local_attr_ancestors(node.name):
            # get astroid for the searched method
            try:
                meth_node = overridden[node.name]
            except KeyError:
                # we have found the method but it's not in the local
                # dictionary.
                # This may happen with astroid build from living objects
                continue
            if not isinstance(meth_node, astroid.FunctionDef):
                continue
            self._check_signature(node, meth_node, "overridden", klass)
            break
        if node.decorators:
            for decorator in node.decorators.nodes:
                if isinstance(decorator, astroid.Attribute) and decorator.attrname in (
                    "getter",
                    "setter",
                    "deleter",
                ):
                    # attribute affectation will call this method, not hiding it
                    return
                if isinstance(decorator, astroid.Name):
                    if decorator.name == "property":
                        # attribute affectation will either call a setter or raise
                        # an attribute error, anyway not hiding the function
                        return

                # Infer the decorator and see if it returns something useful
                inferred = safe_infer(decorator)
                if not inferred:
                    return
                if isinstance(inferred, astroid.FunctionDef):
                    # Okay, it's a decorator, let's see what it can infer.
                    try:
                        inferred = next(inferred.infer_call_result(inferred))
                    except astroid.InferenceError:
                        return
                try:
                    if (
                        isinstance(inferred, (astroid.Instance, astroid.ClassDef))
                        and inferred.getattr("__get__")
                        and inferred.getattr("__set__")
                    ):
                        return
                except astroid.AttributeInferenceError:
                    pass

        # check if the method is hidden by an attribute
        try:
            overridden = klass.instance_attr(node.name)[0]  # XXX
            overridden_frame = overridden.frame()
            if (
                isinstance(overridden_frame, astroid.FunctionDef)
                and overridden_frame.type == "method"
            ):
                overridden_frame = overridden_frame.parent.frame()
            if isinstance(overridden_frame, astroid.ClassDef) and klass.is_subtype_of(
                overridden_frame.qname()
            ):
                args = (overridden.root().name, overridden.fromlineno)
                self.add_message("method-hidden", args=args, node=node)
        except astroid.NotFoundError:
            pass

```

### Rank 3 - `code_1010842` - `_determine_function_name_type`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/base.py`

- Similarity: `0.597977`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/base.py#L305-L340


```python

def _determine_function_name_type(node, config=None):
    """Determine the name type whose regex the a function's name should match.

    :param node: A function node.
    :type node: astroid.node_classes.NodeNG
    :param config: Configuration from which to pull additional property classes.
    :type config: :class:`optparse.Values`

    :returns: One of ('function', 'method', 'attr')
    :rtype: str
    """
    property_classes, property_names = _get_properties(config)
    if not node.is_method():
        return "function"
    if node.decorators:
        decorators = node.decorators.nodes
    else:
        decorators = []
    for decorator in decorators:
        # If the function is a property (decorated with @property
        # or @abc.abstractproperty), the name type is 'attr'.
        if isinstance(decorator, astroid.Name) or (
            isinstance(decorator, astroid.Attribute)
            and decorator.attrname in property_names
        ):
            infered = utils.safe_infer(decorator)
            if infered and infered.qname() in property_classes:
                return "attr"
        # If the function is decorated using the prop_method.{setter,getter}
        # form, treat it like an attribute as well.
        elif isinstance(decorator, astroid.Attribute) and decorator.attrname in (
            "setter",
            "deleter",
        ):
            return "attr"
    return "method"

```

### Rank 4 - `code_1016758` - `_is_attribute_property`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/classes.py`

- Similarity: `0.590468`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/classes.py#L368-L397


```python

def _is_attribute_property(name, klass):
    """ Check if the given attribute *name* is a property
    in the given *klass*.

    It will look for `property` calls or for functions
    with the given name, decorated by `property` or `property`
    subclasses.
    Returns ``True`` if the name is a property in the given klass,
    ``False`` otherwise.
    """

    try:
        attributes = klass.getattr(name)
    except astroid.NotFoundError:
        return False
    property_name = "{}.property".format(BUILTINS)
    for attr in attributes:
        if attr is astroid.Uninferable:
            continue
        try:
            infered = next(attr.infer())
        except astroid.InferenceError:
            continue
        if isinstance(infered, astroid.FunctionDef) and decorated_with_property(
            infered
        ):
            return True
        if infered.pytype() == property_name:
            return True
    return False

```

### Rank 5 - `code_1011754` - `_is_dataclass`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/design_analysis.py`

- Similarity: `0.576100`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/design_analysis.py#L129-L153


```python

def _is_dataclass(node: astroid.ClassDef) -> bool:
    """Check if a class definition defines a Python 3.7+ dataclass

    :param node: The class node to check.
    :type node: astroid.ClassDef

    :returns: True if the given node represents a dataclass class. False otherwise.
    :rtype: bool
    """
    if not node.decorators:
        return False

    root_locals = node.root().locals
    for decorator in node.decorators.nodes:
        if isinstance(decorator, astroid.Call):
            decorator = decorator.func
        if not isinstance(decorator, (astroid.Name, astroid.Attribute)):
            continue
        if isinstance(decorator, astroid.Name):
            name = decorator.name
        else:
            name = decorator.attrname
        if name == DATACLASS_DECORATOR and DATACLASS_DECORATOR in root_locals:
            return True
    return False

```

### Rank 6 - `code_1020000` - `ClassChecker._check_useless_super_delegation`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/classes.py`

- Similarity: `0.555627`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/classes.py#L938-L1042


```python

def _check_useless_super_delegation(self, function):
        """Check if the given function node is an useless method override

        We consider it *useless* if it uses the super() builtin, but having
        nothing additional whatsoever than not implementing the method at all.
        If the method uses super() to delegate an operation to the rest of the MRO,
        and if the method called is the same as the current one, the arguments
        passed to super() are the same as the parameters that were passed to
        this method, then the method could be removed altogether, by letting
        other implementation to take precedence.
        """

        if (
            not function.is_method()
            # With decorators is a change of use
            or function.decorators
        ):
            return

        body = function.body
        if len(body) != 1:
            # Multiple statements, which means this overridden method
            # could do multiple things we are not aware of.
            return

        statement = body[0]
        if not isinstance(statement, (astroid.Expr, astroid.Return)):
            # Doing something else than what we are interested into.
            return

        call = statement.value
        if (
            not isinstance(call, astroid.Call)
            # Not a super() attribute access.
            or not isinstance(call.func, astroid.Attribute)
        ):
            return

        # Should be a super call.
        try:
            super_call = next(call.func.expr.infer())
        except astroid.InferenceError:
            return
        else:
            if not isinstance(super_call, objects.Super):
                return

        # The name should be the same.
        if call.func.attrname != function.name:
            return

        # Should be a super call with the MRO pointer being the
        # current class and the type being the current instance.
        current_scope = function.parent.scope()
        if (
            super_call.mro_pointer != current_scope
            or not isinstance(super_call.type, astroid.Instance)
            or super_call.type.name != current_scope.name
        ):
            return

        #  Check values of default args
        klass = function.parent.frame()
        meth_node = None
        for overridden in klass.local_attr_ancestors(function.name):
            # get astroid for the searched method
            try:
                meth_node = overridden[function.name]
            except KeyError:
                # we have found the method but it's not in the local
                # dictionary.
                # This may happen with astroid build from living objects
                continue
            if (
                not isinstance(meth_node, astroid.FunctionDef)
                # If the method have an ancestor which is not a
                # function then it is legitimate to redefine it
                or _has_different_parameters_default_value(
                    meth_node.args, function.args
                )
            ):
                return
            break

        # Detect if the parameters are the same as the call's arguments.
        params = _signature_from_arguments(function.args)
        args = _signature_from_call(call)

        if meth_node is not None:

            def form_annotations(annotations):
                return [
                    annotation.as_string() for annotation in filter(None, annotations)
                ]

            called_annotations = form_annotations(function.args.annotations)
            overridden_annotations = form_annotations(meth_node.args.annotations)
            if called_annotations and overridden_annotations:
                if called_annotations != overridden_annotations:
                    return

        if _definition_equivalent_to_call(params, args):
            self.add_message(
                "useless-super-delegation", node=function, args=(function.name,)
            )

```

### Rank 7 - `code_1026183` - `TypeChecker.visit_assign`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/typecheck.py`

- Similarity: `0.553561`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/typecheck.py#L948-L983


```python

def visit_assign(self, node):
        """check that if assigning to a function call, the function is
        possibly returning something valuable
        """
        if not isinstance(node.value, astroid.Call):
            return
        function_node = safe_infer(node.value.func)
        # skip class, generator and incomplete function definition
        funcs = (astroid.FunctionDef, astroid.UnboundMethod, astroid.BoundMethod)
        if not (
            isinstance(function_node, funcs)
            and function_node.root().fully_defined()
            and not function_node.decorators
        ):
            return
        if (
            function_node.is_generator()
            or function_node.is_abstract(pass_is_abstract=False)
            or isinstance(function_node, astroid.AsyncFunctionDef)
        ):
            return
        returns = list(
            function_node.nodes_of_class(astroid.Return, skip_klass=astroid.FunctionDef)
        )
        if not returns:
            self.add_message("assignment-from-no-return", node=node)
        else:
            for rnode in returns:
                if not (
                    isinstance(rnode.value, astroid.Const)
                    and rnode.value.value is None
                    or rnode.value is None
                ):
                    break
            else:
                self.add_message("assignment-from-none", node=node)

```

### Rank 8 - `code_1030899` - `check_messages`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/utils.py`

- Similarity: `0.549231`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/utils.py#L430-L437


```python

def check_messages(*messages: str) -> Callable:
    """decorator to store messages that are handled by a checker method"""

    def store_messages(func):
        func.checks_msgs = messages
        return func

    return store_messages

```

### Rank 9 - `code_1034730` - `is_registered_in_singledispatch_function`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/utils.py`

- Similarity: `0.536361`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/utils.py#L1128-L1158


```python

def is_registered_in_singledispatch_function(node: astroid.FunctionDef) -> bool:
    """Check if the given function node is a singledispatch function."""

    singledispatch_qnames = (
        "functools.singledispatch",
        "singledispatch.singledispatch",
    )

    if not isinstance(node, astroid.FunctionDef):
        return False

    decorators = node.decorators.nodes if node.decorators else []
    for decorator in decorators:
        # func.register are function calls
        if not isinstance(decorator, astroid.Call):
            continue

        func = decorator.func
        if not isinstance(func, astroid.Attribute) or func.attrname != "register":
            continue

        try:
            func_def = next(func.expr.infer())
        except astroid.InferenceError:
            continue

        if isinstance(func_def, astroid.FunctionDef):
            # pylint: disable=redundant-keyword-arg; some flow inference goes wrong here
            return decorated_with(func_def, singledispatch_qnames)

    return False

```

### Rank 10 - `code_1042625` - `ClassChecker._check_classmethod_declaration`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/classes.py`

- Similarity: `0.522766`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/classes.py#L1202-L1239


```python

def _check_classmethod_declaration(self, node):
        """Checks for uses of classmethod() or staticmethod()

        When a @classmethod or @staticmethod decorator should be used instead.
        A message will be emitted only if the assignment is at a class scope
        and only if the classmethod's argument belongs to the class where it
        is defined.
        `node` is an assign node.
        """
        if not isinstance(node.value, astroid.Call):
            return

        # check the function called is "classmethod" or "staticmethod"
        func = node.value.func
        if not isinstance(func, astroid.Name) or func.name not in (
            "classmethod",
            "staticmethod",
        ):
            return

        msg = (
            "no-classmethod-decorator"
            if func.name == "classmethod"
            else "no-staticmethod-decorator"
        )
        # assignment must be at a class scope
        parent_class = node.scope()
        if not isinstance(parent_class, astroid.ClassDef):
            return

        # Check if the arg passed to classmethod is a class member
        classmeth_arg = node.value.args[0]
        if not isinstance(classmeth_arg, astroid.Name):
            return

        method_name = classmeth_arg.name
        if any(method_name == member.name for member in parent_class.mymethods()):
            self.add_message(msg, node=node.targets[0])

```

### Rank 11 - `code_1035583` - `BasicChecker.visit_functiondef`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/base.py`

- Similarity: `0.520584`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/base.py#L1214-L1219


```python

def visit_functiondef(self, node):
        """check function name, docstring, arguments, redefinition,
        variable names, max locals
        """
        self.stats[node.is_method() and "method" or "function"] += 1
        self._check_dangerous_default(node)

```

### Rank 12 - `code_1040139` - `is_attr_protected`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/utils.py`

- Similarity: `0.509303`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/utils.py#L603-L611


```python

def is_attr_protected(attrname: str) -> bool:
    """return True if attribute name is protected (start with _ and some other
    details), False otherwise.
    """
    return (
        attrname[0] == "_"
        and attrname != "_"
        and not (attrname.startswith("__") and attrname.endswith("__"))
    )

```

### Rank 13 - `code_1038572` - `is_method_call`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/logging.py`

- Similarity: `0.506171`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/logging.py#L99-L116


```python

def is_method_call(func, types=(), methods=()):
    """Determines if a BoundMethod node represents a method call.

    Args:
      func (astroid.BoundMethod): The BoundMethod AST node to check.
      types (Optional[String]): Optional sequence of caller type names to restrict check.
      methods (Optional[String]): Optional sequence of method names to restrict check.

    Returns:
      bool: true if the node represents a method call for the given type and
      method names, False otherwise.
    """
    return (
        isinstance(func, astroid.BoundMethod)
        and isinstance(func.bound, astroid.Instance)
        and (func.bound.name in types if types else True)
        and (func.name in methods if methods else True)
    )

```

### Rank 14 - `code_1039906` - `is_node_inside_try_except`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/utils.py`

- Similarity: `0.504750`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/utils.py#L877-L888


```python

def is_node_inside_try_except(node: astroid.Raise) -> bool:
    """Check if the node is directly under a Try/Except statement.
    (but not under an ExceptHandler!)

    Args:
        node (astroid.Raise): the node raising the exception.

    Returns:
        bool: True if the node is inside a try/except statement, False otherwise.
    """
    context = find_try_except_wrapper_node(node)
    return isinstance(context, astroid.TryExcept)

```

### Rank 15 - `code_1033231` - `get_setters_property_name`

- Repository: `PyCQA/pylint`

- Path: `pylint/extensions/_check_docs_utils.py`

- Similarity: `0.503000`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/extensions/_check_docs_utils.py#L39-L57


```python

def get_setters_property_name(node):
    """Get the name of the property that the given node is a setter for.

    :param node: The node to get the property name for.
    :type node: str

    :rtype: str or None
    :returns: The name of the property that the node is a setter for,
        or None if one could not be found.
    """
    decorators = node.decorators.nodes if node.decorators else []
    for decorator in decorators:
        if (
            isinstance(decorator, astroid.Attribute)
            and decorator.attrname == "setter"
            and isinstance(decorator.expr, astroid.Name)
        ):
            return decorator.expr.name
    return None

```

### Rank 16 - `code_1026225` - `overrides_a_method`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/utils.py`

- Similarity: `0.500123`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/utils.py#L422-L427


```python

def overrides_a_method(class_node: astroid.node_classes.NodeNG, name: str) -> bool:
    """return True if <name> is a method overridden from an ancestor"""
    for ancestor in class_node.ancestors():
        if name in ancestor and isinstance(ancestor[name], astroid.FunctionDef):
            return True
    return False

```

### Rank 17 - `code_1025158` - `is_func_decorator`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/utils.py`

- Similarity: `0.500074`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/utils.py#L383-L395


```python

def is_func_decorator(node: astroid.node_classes.NodeNG) -> bool:
    """return true if the name is used in function decorator"""
    parent = node.parent
    while parent is not None:
        if isinstance(parent, astroid.Decorators):
            return True
        if parent.is_statement or isinstance(
            parent,
            (astroid.Lambda, scoped_nodes.ComprehensionScope, scoped_nodes.ListComp),
        ):
            break
        parent = parent.parent
    return False

```

### Rank 18 - `code_1034778` - `BasicChecker.visit_expr`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/base.py`

- Similarity: `0.488438`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/base.py#L1081-L1124


```python

def visit_expr(self, node):
        """check for various kind of statements without effect"""
        expr = node.value
        if isinstance(expr, astroid.Const) and isinstance(expr.value, str):
            # treat string statement in a separated message
            # Handle PEP-257 attribute docstrings.
            # An attribute docstring is defined as being a string right after
            # an assignment at the module level, class level or __init__ level.
            scope = expr.scope()
            if isinstance(
                scope, (astroid.ClassDef, astroid.Module, astroid.FunctionDef)
            ):
                if isinstance(scope, astroid.FunctionDef) and scope.name != "__init__":
                    pass
                else:
                    sibling = expr.previous_sibling()
                    if (
                        sibling is not None
                        and sibling.scope() is scope
                        and isinstance(sibling, (astroid.Assign, astroid.AnnAssign))
                    ):
                        return
            self.add_message("pointless-string-statement", node=node)
            return

        # Ignore if this is :
        # * a direct function call
        # * the unique child of a try/except body
        # * a yieldd statement
        # * an ellipsis (which can be used on Python 3 instead of pass)
        # warn W0106 if we have any underlying function call (we can't predict
        # side effects), else pointless-statement
        if isinstance(
            expr, (astroid.Yield, astroid.Await, astroid.Ellipsis, astroid.Call)
        ) or (
            isinstance(node.parent, astroid.TryExcept) and node.parent.body == [node]
        ):
            return
        if any(expr.nodes_of_class(astroid.Call)):
            self.add_message(
                "expression-not-assigned", node=node, args=expr.as_string()
            )
        else:
            self.add_message("pointless-statement", node=node)

```

### Rank 19 - `code_1043790` - `clobber_in_except`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/utils.py`

- Similarity: `0.488347`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/utils.py#L240-L262


```python

def clobber_in_except(
    node: astroid.node_classes.NodeNG
) -> Tuple[bool, Tuple[str, str]]:
    """Checks if an assignment node in an except handler clobbers an existing
    variable.

    Returns (True, args for W0623) if assignment clobbers an existing variable,
    (False, None) otherwise.
    """
    if isinstance(node, astroid.AssignAttr):
        return True, (node.attrname, "object %r" % (node.expr.as_string(),))
    if isinstance(node, astroid.AssignName):
        name = node.name
        if is_builtin(name):
            return (True, (name, "builtins"))

        stmts = node.lookup(name)[1]
        if stmts and not isinstance(
            stmts[0].assign_type(),
            (astroid.Assign, astroid.AugAssign, astroid.ExceptHandler),
        ):
            return True, (name, "outer scope (line %s)" % stmts[0].fromlineno)
    return False, None

```

### Rank 20 - `code_1024098` - `is_default_argument`

- Repository: `PyCQA/pylint`

- Path: `pylint/checkers/utils.py`

- Similarity: `0.487779`

- Source URL: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/utils.py#L370-L380


```python

def is_default_argument(node: astroid.node_classes.NodeNG) -> bool:
    """return true if the given Name node is used in function or lambda
    default argument's value
    """
    parent = node.scope()
    if isinstance(parent, (astroid.FunctionDef, astroid.Lambda)):
        for default_node in parent.args.defaults:
            for default_name_node in default_node.nodes_of_class(astroid.Name):
                if default_name_node is node:
                    return True
    return False

```

## csn_42 - Cloud SQL delete completion

Candidates exported: 20

### Rank 1 - `code_1016745` - `SQLUtility.destroy`

- Repository: `memsql/memsql-python`

- Path: `memsql/common/sql_utility.py`

- Similarity: `0.883587`

- Source URL: https://github.com/memsql/memsql-python/blob/aac223a1b937d5b348b42af3c601a6c685ca633a/memsql/common/sql_utility.py#L34-L39


```python

def destroy(self):
        """ Destroy the SQLStepQueue tables in the database """
        with self._db_conn() as conn:
            for table_name in self._tables:
                conn.execute('DROP TABLE IF EXISTS %s' % table_name)
        return self

```

### Rank 2 - `code_1012819` - `AzureCosmosDBHook.delete_database`

- Repository: `apache/airflow`

- Path: `airflow/contrib/hooks/azure_cosmos_hook.py`

- Similarity: `0.875000`

- Source URL: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/contrib/hooks/azure_cosmos_hook.py#L162-L169


```python

def delete_database(self, database_name):
        """
        Deletes an existing database in CosmosDB.
        """
        if database_name is None:
            raise AirflowBadRequest("Database name cannot be None.")

        self.get_conn().DeleteDatabase(get_database_link(database_name))

```

### Rank 3 - `code_1002485` - `CloudSpannerHook.delete_database`

- Repository: `apache/airflow`

- Path: `airflow/contrib/hooks/gcp_spanner_hook.py`

- Similarity: `0.864840`

- Source URL: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/contrib/hooks/gcp_spanner_hook.py#L291-L325


```python

def delete_database(self, instance_id, database_id, project_id=None):
        """
        Drops a database in Cloud Spanner.

        :type project_id: str
        :param instance_id: The ID of the Cloud Spanner instance.
        :type instance_id: str
        :param database_id: The ID of the database in Cloud Spanner.
        :type database_id: str
        :param project_id: Optional, the ID of the  GCP project that owns the Cloud Spanner
            database. If set to None or missing, the default project_id from the GCP connection is used.
        :return: True if everything succeeded
        :rtype: bool
        """

        instance = self._get_client(project_id=project_id).\
            instance(instance_id=instance_id)
        if not instance.exists():
            raise AirflowException("The instance {} does not exist in project {} !".
                                   format(instance_id, project_id))
        database = instance.database(database_id=database_id)
        if not database.exists():
            self.log.info("The database {} is already deleted from instance {}. "
                          "Exiting.".format(database_id, instance_id))
            return
        try:
            operation = database.drop()  # type: Operation
        except GoogleAPICallError as e:
            self.log.error('An error occurred: %s. Exiting.', e.message)
            raise e

        if operation:
            result = operation.result()
            self.log.info(result)
        return

```

### Rank 4 - `code_1023446` - `AzureCosmosDBHook.delete_collection`

- Repository: `apache/airflow`

- Path: `airflow/contrib/hooks/azure_cosmos_hook.py`

- Similarity: `0.817407`

- Source URL: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/contrib/hooks/azure_cosmos_hook.py#L171-L179


```python

def delete_collection(self, collection_name, database_name=None):
        """
        Deletes an existing collection in the CosmosDB database.
        """
        if collection_name is None:
            raise AirflowBadRequest("Collection name cannot be None.")

        self.get_conn().DeleteContainer(
            get_collection_link(self.__get_database_name(database_name), collection_name))

```

### Rank 5 - `code_1026579` - `BigtableHook.delete_table`

- Repository: `apache/airflow`

- Path: `airflow/contrib/hooks/gcp_bigtable_hook.py`

- Similarity: `0.767614`

- Source URL: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/contrib/hooks/gcp_bigtable_hook.py#L199-L214


```python

def delete_table(self, instance_id, table_id, project_id=None):
        """
        Deletes the specified table in Cloud Bigtable.
        Raises google.api_core.exceptions.NotFound if the table does not exist.

        :type instance_id: str
        :param instance_id: The ID of the Cloud Bigtable instance.
        :type table_id: str
        :param table_id: The ID of the table in Cloud Bigtable.
        :type project_id: str
        :param project_id: Optional, Google Cloud Platform project ID where the
            BigTable exists. If set to None or missing,
            the default project_id from the GCP connection is used.
        """
        table = self.get_instance(instance_id=instance_id, project_id=project_id).table(table_id=table_id)
        table.delete()

```

### Rank 6 - `code_1026572` - `AzureCosmosDBHook.delete_document`

- Repository: `apache/airflow`

- Path: `airflow/contrib/hooks/azure_cosmos_hook.py`

- Similarity: `0.723483`

- Source URL: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/contrib/hooks/azure_cosmos_hook.py#L226-L237


```python

def delete_document(self, document_id, database_name=None, collection_name=None):
        """
        Delete an existing document out of a collection in the CosmosDB database.
        """
        if document_id is None:
            raise AirflowBadRequest("Cannot delete a document without an id")

        self.get_conn().DeleteItem(
            get_document_link(
                self.__get_database_name(database_name),
                self.__get_collection_name(collection_name),
                document_id))

```

### Rank 7 - `code_1001755` - `CloudSqlHook.export_instance`

- Repository: `apache/airflow`

- Path: `airflow/contrib/hooks/gcp_sql_hook.py`

- Similarity: `0.716563`

- Source URL: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/contrib/hooks/gcp_sql_hook.py#L282-L310


```python

def export_instance(self, instance, body, project_id=None):
        """
        Exports data from a Cloud SQL instance to a Cloud Storage bucket as a SQL dump
        or CSV file.

        :param instance: Database instance ID of the Cloud SQL instance. This does not include the
            project ID.
        :type instance: str
        :param body: The request body, as described in
            https://cloud.google.com/sql/docs/mysql/admin-api/v1beta4/instances/export#request-body
        :type body: dict
        :param project_id: Project ID of the project that contains the instance. If set
            to None or missing, the default project_id from the GCP connection is used.
        :type project_id: str
        :return: None
        """
        try:
            response = self.get_conn().instances().export(
                project=project_id,
                instance=instance,
                body=body
            ).execute(num_retries=self.num_retries)
            operation_name = response["name"]
            self._wait_for_operation_to_complete(project_id=project_id,
                                                 operation_name=operation_name)
        except HttpError as ex:
            raise AirflowException(
                'Exporting instance {} failed: {}'.format(instance, ex.content)
            )

```

### Rank 8 - `code_1011288` - `CloudSqlHook.create_database`

- Repository: `apache/airflow`

- Path: `airflow/contrib/hooks/gcp_sql_hook.py`

- Similarity: `0.716157`

- Source URL: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/contrib/hooks/gcp_sql_hook.py#L205-L226


```python

def create_database(self, instance, body, project_id=None):
        """
        Creates a new database inside a Cloud SQL instance.

        :param instance: Database instance ID. This does not include the project ID.
        :type instance: str
        :param body: The request body, as described in
            https://cloud.google.com/sql/docs/mysql/admin-api/v1beta4/databases/insert#request-body.
        :type body: dict
        :param project_id: Project ID of the project that contains the instance. If set
            to None or missing, the default project_id from the GCP connection is used.
        :type project_id: str
        :return: None
        """
        response = self.get_conn().databases().insert(
            project=project_id,
            instance=instance,
            body=body
        ).execute(num_retries=self.num_retries)
        operation_name = response["name"]
        self._wait_for_operation_to_complete(project_id=project_id,
                                             operation_name=operation_name)

```

### Rank 9 - `code_1013890` - `CloudSqlHook.patch_database`

- Repository: `apache/airflow`

- Path: `airflow/contrib/hooks/gcp_sql_hook.py`

- Similarity: `0.714993`

- Source URL: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/contrib/hooks/gcp_sql_hook.py#L229-L256


```python

def patch_database(self, instance, database, body, project_id=None):
        """
        Updates a database resource inside a Cloud SQL instance.

        This method supports patch semantics.
        See https://cloud.google.com/sql/docs/mysql/admin-api/how-tos/performance#patch.

        :param instance: Database instance ID. This does not include the project ID.
        :type instance: str
        :param database: Name of the database to be updated in the instance.
        :type database: str
        :param body: The request body, as described in
            https://cloud.google.com/sql/docs/mysql/admin-api/v1beta4/databases/insert#request-body.
        :type body: dict
        :param project_id: Project ID of the project that contains the instance. If set
            to None or missing, the default project_id from the GCP connection is used.
        :type project_id: str
        :return: None
        """
        response = self.get_conn().databases().patch(
            project=project_id,
            instance=instance,
            database=database,
            body=body
        ).execute(num_retries=self.num_retries)
        operation_name = response["name"]
        self._wait_for_operation_to_complete(project_id=project_id,
                                             operation_name=operation_name)

```

### Rank 10 - `code_1025778` - `CloudSqlHook.get_instance`

- Repository: `apache/airflow`

- Path: `airflow/contrib/hooks/gcp_sql_hook.py`

- Similarity: `0.710863`

- Source URL: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/contrib/hooks/gcp_sql_hook.py#L97-L112


```python

def get_instance(self, instance, project_id=None):
        """
        Retrieves a resource containing information about a Cloud SQL instance.

        :param instance: Database instance ID. This does not include the project ID.
        :type instance: str
        :param project_id: Project ID of the project that contains the instance. If set
            to None or missing, the default project_id from the GCP connection is used.
        :type project_id: str
        :return: A Cloud SQL instance resource.
        :rtype: dict
        """
        return self.get_conn().instances().get(
            project=project_id,
            instance=instance
        ).execute(num_retries=self.num_retries)

```

### Rank 11 - `code_1027146` - `CloudSqlHook.get_database`

- Repository: `apache/airflow`

- Path: `airflow/contrib/hooks/gcp_sql_hook.py`

- Similarity: `0.693314`

- Source URL: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/contrib/hooks/gcp_sql_hook.py#L183-L202


```python

def get_database(self, instance, database, project_id=None):
        """
        Retrieves a database resource from a Cloud SQL instance.

        :param instance: Database instance ID. This does not include the project ID.
        :type instance: str
        :param database: Name of the database in the instance.
        :type database: str
        :param project_id: Project ID of the project that contains the instance. If set
            to None or missing, the default project_id from the GCP connection is used.
        :type project_id: str
        :return: A Cloud SQL database resource, as described in
            https://cloud.google.com/sql/docs/mysql/admin-api/v1beta4/databases#resource.
        :rtype: dict
        """
        return self.get_conn().databases().get(
            project=project_id,
            instance=instance,
            database=database
        ).execute(num_retries=self.num_retries)

```

### Rank 12 - `code_1037762` - `CloudSqlHook.patch_instance`

- Repository: `apache/airflow`

- Path: `airflow/contrib/hooks/gcp_sql_hook.py`

- Similarity: `0.687432`

- Source URL: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/contrib/hooks/gcp_sql_hook.py#L136-L160


```python

def patch_instance(self, body, instance, project_id=None):
        """
        Updates settings of a Cloud SQL instance.

        Caution: This is not a partial update, so you must include values for
        all the settings that you want to retain.

        :param body: Body required by the Cloud SQL patch API, as described in
            https://cloud.google.com/sql/docs/mysql/admin-api/v1beta4/instances/patch#request-body.
        :type body: dict
        :param instance: Cloud SQL instance ID. This does not include the project ID.
        :type instance: str
        :param project_id: Project ID of the project that contains the instance. If set
            to None or missing, the default project_id from the GCP connection is used.
        :type project_id: str
        :return: None
        """
        response = self.get_conn().instances().patch(
            project=project_id,
            instance=instance,
            body=body
        ).execute(num_retries=self.num_retries)
        operation_name = response["name"]
        self._wait_for_operation_to_complete(project_id=project_id,
                                             operation_name=operation_name)

```

### Rank 13 - `code_1041412` - `CloudSqlHook.create_instance`

- Repository: `apache/airflow`

- Path: `airflow/contrib/hooks/gcp_sql_hook.py`

- Similarity: `0.687134`

- Source URL: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/contrib/hooks/gcp_sql_hook.py#L115-L133


```python

def create_instance(self, body, project_id=None):
        """
        Creates a new Cloud SQL instance.

        :param body: Body required by the Cloud SQL insert API, as described in
            https://cloud.google.com/sql/docs/mysql/admin-api/v1beta4/instances/insert#request-body.
        :type body: dict
        :param project_id: Project ID of the project that contains the instance. If set
            to None or missing, the default project_id from the GCP connection is used.
        :type project_id: str
        :return: None
        """
        response = self.get_conn().instances().insert(
            project=project_id,
            body=body
        ).execute(num_retries=self.num_retries)
        operation_name = response["name"]
        self._wait_for_operation_to_complete(project_id=project_id,
                                             operation_name=operation_name)

```

### Rank 14 - `code_1004155` - `CloudSqlDatabaseHook.delete_connection`

- Repository: `apache/airflow`

- Path: `airflow/contrib/hooks/gcp_sql_hook.py`

- Similarity: `0.685972`

- Source URL: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/contrib/hooks/gcp_sql_hook.py#L926-L941


```python

def delete_connection(self, session=None):
        """
        Delete the dynamically created connection from the Connection table.

        :param session: Session of the SQL Alchemy ORM (automatically generated with
                        decorator).
        """
        self.log.info("Deleting connection %s", self.db_conn_id)
        connections = session.query(Connection).filter(
            Connection.conn_id == self.db_conn_id)
        if connections.count():
            connection = connections[0]
            session.delete(connection)
            session.commit()
        else:
            self.log.info("Connection was already deleted!")

```

### Rank 15 - `code_1006706` - `CloudSqlHook.delete_instance`

- Repository: `apache/airflow`

- Path: `airflow/contrib/hooks/gcp_sql_hook.py`

- Similarity: `0.682094`

- Source URL: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/contrib/hooks/gcp_sql_hook.py#L163-L180


```python

def delete_instance(self, instance, project_id=None):
        """
        Deletes a Cloud SQL instance.

        :param project_id: Project ID of the project that contains the instance. If set
            to None or missing, the default project_id from the GCP connection is used.
        :type project_id: str
        :param instance: Cloud SQL instance ID. This does not include the project ID.
        :type instance: str
        :return: None
        """
        response = self.get_conn().instances().delete(
            project=project_id,
            instance=instance,
        ).execute(num_retries=self.num_retries)
        operation_name = response["name"]
        self._wait_for_operation_to_complete(project_id=project_id,
                                             operation_name=operation_name)

```

### Rank 16 - `code_1004389` - `CloudSqlDatabaseHook.cleanup_database_hook`

- Repository: `apache/airflow`

- Path: `airflow/contrib/hooks/gcp_sql_hook.py`

- Similarity: `0.680885`

- Source URL: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/contrib/hooks/gcp_sql_hook.py#L974-L982


```python

def cleanup_database_hook(self):
        """
        Clean up database hook after it was used.
        """
        if self.database_type == 'postgres':
            if hasattr(self.db_hook,
                       'conn') and self.db_hook.conn and self.db_hook.conn.notices:
                for output in self.db_hook.conn.notices:
                    self.log.info(output)

```

### Rank 17 - `code_1010213` - `CloudSqlDatabaseHook.retrieve_connection`

- Repository: `apache/airflow`

- Path: `airflow/contrib/hooks/gcp_sql_hook.py`

- Similarity: `0.680857`

- Source URL: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/contrib/hooks/gcp_sql_hook.py#L911-L923


```python

def retrieve_connection(self, session=None):
        """
        Retrieves the dynamically created connection from the Connection table.

        :param session: Session of the SQL Alchemy ORM (automatically generated with
                        decorator).
        """
        self.log.info("Retrieving connection %s", self.db_conn_id)
        connections = session.query(Connection).filter(
            Connection.conn_id == self.db_conn_id)
        if connections.count():
            return connections[0]
        return None

```

### Rank 18 - `code_1010995` - `CloudSqlDatabaseHook.get_sqlproxy_runner`

- Repository: `apache/airflow`

- Path: `airflow/contrib/hooks/gcp_sql_hook.py`

- Similarity: `0.679004`

- Source URL: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/contrib/hooks/gcp_sql_hook.py#L943-L959


```python

def get_sqlproxy_runner(self):
        """
        Retrieve Cloud SQL Proxy runner. It is used to manage the proxy
        lifecycle per task.

        :return: The Cloud SQL Proxy runner.
        :rtype: CloudSqlProxyRunner
        """
        if not self.use_proxy:
            raise AirflowException("Proxy runner can only be retrieved in case of use_proxy = True")
        return CloudSqlProxyRunner(
            path_prefix=self.sql_proxy_unique_path,
            instance_specification=self._get_sqlproxy_instance_specification(),
            project_id=self.project_id,
            sql_proxy_version=self.sql_proxy_version,
            sql_proxy_binary_path=self.sql_proxy_binary_path
        )

```

### Rank 19 - `code_1024867` - `CloudSqlDatabaseHook.reserve_free_tcp_port`

- Repository: `apache/airflow`

- Path: `airflow/contrib/hooks/gcp_sql_hook.py`

- Similarity: `0.675905`

- Source URL: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/contrib/hooks/gcp_sql_hook.py#L984-L990


```python

def reserve_free_tcp_port(self):
        """
        Reserve free TCP port to be used by Cloud SQL Proxy
        """
        self.reserved_tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.reserved_tcp_socket.bind(('127.0.0.1', 0))
        self.sql_proxy_tcp_port = self.reserved_tcp_socket.getsockname()[1]

```

### Rank 20 - `code_1013886` - `CloudSqlDatabaseHook.get_database_hook`

- Repository: `apache/airflow`

- Path: `airflow/contrib/hooks/gcp_sql_hook.py`

- Similarity: `0.671000`

- Source URL: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/contrib/hooks/gcp_sql_hook.py#L961-L972


```python

def get_database_hook(self):
        """
        Retrieve database hook. This is the actual Postgres or MySQL database hook
        that uses proxy or connects directly to the Google Cloud SQL database.
        """
        if self.database_type == 'postgres':
            self.db_hook = PostgresHook(postgres_conn_id=self.db_conn_id,
                                        schema=self.database)
        else:
            self.db_hook = MySqlHook(mysql_conn_id=self.db_conn_id,
                                     schema=self.database)
        return self.db_hook

```

## csn_9388 - URL query removal

Candidates exported: 20

### Rank 1 - `code_1012695` - `Zotero._build_query`

- Repository: `urschrei/pyzotero`

- Path: `pyzotero/zotero.py`

- Similarity: `0.704000`

- Source URL: https://github.com/urschrei/pyzotero/blob/b378966b30146a952f7953c23202fb5a1ddf81d9/pyzotero/zotero.py#L429-L443


```python

def _build_query(self, query_string, no_params=False):
        """
        Set request parameters. Will always add the user ID if it hasn't
        been specifically set by an API method
        """
        try:
            query = quote(query_string.format(u=self.library_id, t=self.library_type))
        except KeyError as err:
            raise ze.ParamNotPassed("There's a request parameter missing: %s" % err)
        # Add the URL parameters and the user key, if necessary
        if no_params is False:
            if not self.url_params:
                self.add_parameters()
            query = "%s?%s" % (query, self.url_params)
        return query

```

### Rank 2 - `code_1003669` - `Provider._make_response`

- Repository: `NateFerrero/oauth2lib`

- Path: `oauth2lib/provider.py`

- Similarity: `0.679318`

- Source URL: https://github.com/NateFerrero/oauth2lib/blob/d161b010f8a596826050a09e5e94d59443cc12d9/oauth2lib/provider.py#L24-L40


```python

def _make_response(self, body='', headers=None, status_code=200):
        """Return a response object from the given parameters.

        :param body: Buffer/string containing the response body.
        :type body: str
        :param headers: Dict of headers to include in the requests.
        :type headers: dict
        :param status_code: HTTP status code.
        :type status_code: int
        :rtype: requests.Response
        """
        res = Response()
        res.status_code = status_code
        if headers is not None:
            res.headers.update(headers)
        res.raw = StringIO(body)
        return res

```

### Rank 3 - `code_1004447` - `Client.get_authorization_code_uri`

- Repository: `NateFerrero/oauth2lib`

- Path: `oauth2lib/client.py`

- Similarity: `0.674671`

- Source URL: https://github.com/NateFerrero/oauth2lib/blob/d161b010f8a596826050a09e5e94d59443cc12d9/oauth2lib/client.py#L49-L60


```python

def get_authorization_code_uri(self, **params):
        """Construct a full URL that can be used to obtain an authorization
        code from the provider authorization_uri. Use this URI in a client
        frame to cause the provider to generate an authorization code.

        :rtype: str
        """
        if 'response_type' not in params:
            params['response_type'] = self.default_response_type
        params.update({'client_id': self.client_id,
                       'redirect_uri': self.redirect_uri})
        return utils.build_url(self.authorization_uri, params)

```

### Rank 4 - `code_1009156` - `Provider._make_json_response`

- Repository: `NateFerrero/oauth2lib`

- Path: `oauth2lib/provider.py`

- Similarity: `0.668000`

- Source URL: https://github.com/NateFerrero/oauth2lib/blob/d161b010f8a596826050a09e5e94d59443cc12d9/oauth2lib/provider.py#L61-L80


```python

def _make_json_response(self, data, headers=None, status_code=200):
        """Return a response object from the given JSON data.

        :param data: Data to JSON-encode.
        :type data: mixed
        :param headers: Dict of headers to include in the requests.
        :type headers: dict
        :param status_code: HTTP status code.
        :type status_code: int
        :rtype: requests.Response
        """
        response_headers = {}
        if headers is not None:
            response_headers.update(headers)
        response_headers['Content-Type'] = 'application/json;charset=UTF-8'
        response_headers['Cache-Control'] = 'no-store'
        response_headers['Pragma'] = 'no-cache'
        return self._make_response(json.dumps(data),
                                   response_headers,
                                   status_code)

```

### Rank 5 - `code_1018012` - `Client.http_post`

- Repository: `NateFerrero/oauth2lib`

- Path: `oauth2lib/client.py`

- Similarity: `0.656203`

- Source URL: https://github.com/NateFerrero/oauth2lib/blob/d161b010f8a596826050a09e5e94d59443cc12d9/oauth2lib/client.py#L36-L47


```python

def http_post(self, url, data=None):
        """POST to URL and get result as a response object.

        :param url: URL to POST.
        :type url: str
        :param data: Data to send in the form body.
        :type data: str
        :rtype: requests.Response
        """
        if not url.startswith('https://'):
            raise ValueError('Protocol must be HTTPS, invalid URL: %s' % url)
        return requests.post(url, data, verify=True)

```

### Rank 6 - `code_1019841` - `AuthorizationProvider.get_token`

- Repository: `NateFerrero/oauth2lib`

- Path: `oauth2lib/provider.py`

- Similarity: `0.653819`

- Source URL: https://github.com/NateFerrero/oauth2lib/blob/d161b010f8a596826050a09e5e94d59443cc12d9/oauth2lib/provider.py#L338-L410


```python

def get_token(self,
                  grant_type,
                  client_id,
                  client_secret,
                  redirect_uri,
                  code,
                  **params):
        """Generate access token HTTP response.

        :param grant_type: Desired grant type. Must be "authorization_code".
        :type grant_type: str
        :param client_id: Client ID.
        :type client_id: str
        :param client_secret: Client secret.
        :type client_secret: str
        :param redirect_uri: Client redirect URI.
        :type redirect_uri: str
        :param code: Authorization code.
        :type code: str
        :rtype: requests.Response
        """

        # Ensure proper grant_type
        if grant_type != 'authorization_code':
            return self._make_json_error_response('unsupported_grant_type')

        # Check conditions
        is_valid_client_id = self.validate_client_id(client_id)
        is_valid_client_secret = self.validate_client_secret(client_id,
                                                             client_secret)
        is_valid_redirect_uri = self.validate_redirect_uri(client_id,
                                                           redirect_uri)

        scope = params.get('scope', '')
        is_valid_scope = self.validate_scope(client_id, scope)
        data = self.from_authorization_code(client_id, code, scope)
        is_valid_grant = data is not None

        # Return proper error responses on invalid conditions
        if not (is_valid_client_id and is_valid_client_secret):
            return self._make_json_error_response('invalid_client')

        if not is_valid_grant or not is_valid_redirect_uri:
            return self._make_json_error_response('invalid_grant')

        if not is_valid_scope:
            return self._make_json_error_response('invalid_scope')

        # Discard original authorization code
        self.discard_authorization_code(client_id, code)

        # Generate access tokens once all conditions have been met
        access_token = self.generate_access_token()
        token_type = self.token_type
        expires_in = self.token_expires_in
        refresh_token = self.generate_refresh_token()

        # Save information to be used to validate later requests
        self.persist_token_information(client_id=client_id,
                                       scope=scope,
                                       access_token=access_token,
                                       token_type=token_type,
                                       expires_in=expires_in,
                                       refresh_token=refresh_token,
                                       data=data)

        # Return json response
        return self._make_json_response({
            'access_token': access_token,
            'token_type': token_type,
            'expires_in': expires_in,
            'refresh_token': refresh_token
        })

```

### Rank 7 - `code_1024170` - `url_query_params`

- Repository: `NateFerrero/oauth2lib`

- Path: `oauth2lib/utils.py`

- Similarity: `0.647735`

- Source URL: https://github.com/NateFerrero/oauth2lib/blob/d161b010f8a596826050a09e5e94d59443cc12d9/oauth2lib/utils.py#L15-L22


```python

def url_query_params(url):
    """Return query parameters as a dict from the specified URL.

    :param url: URL.
    :type url: str
    :rtype: dict
    """
    return dict(urlparse.parse_qsl(urlparse.urlparse(url).query, True))

```

### Rank 8 - `code_1028168` - `AuthorizationProvider.get_authorization_code_from_uri`

- Repository: `NateFerrero/oauth2lib`

- Path: `oauth2lib/provider.py`

- Similarity: `0.634194`

- Source URL: https://github.com/NateFerrero/oauth2lib/blob/d161b010f8a596826050a09e5e94d59443cc12d9/oauth2lib/provider.py#L412-L449


```python

def get_authorization_code_from_uri(self, uri):
        """Get authorization code response from a URI. This method will
        ignore the domain and path of the request, instead
        automatically parsing the query string parameters.

        :param uri: URI to parse for authorization information.
        :type uri: str
        :rtype: requests.Response
        """
        params = utils.url_query_params(uri)
        try:
            if 'response_type' not in params:
                raise TypeError('Missing parameter response_type in URL query')

            if 'client_id' not in params:
                raise TypeError('Missing parameter client_id in URL query')

            if 'redirect_uri' not in params:
                raise TypeError('Missing parameter redirect_uri in URL query')

            return self.get_authorization_code(**params)
        except TypeError as exc:
            self._handle_exception(exc)

            # Catch missing parameters in request
            err = 'invalid_request'
            if 'redirect_uri' in params:
                u = params['redirect_uri']
                return self._make_redirect_error_response(u, err)
            else:
                return self._invalid_redirect_uri_response()
        except StandardError as exc:
            self._handle_exception(exc)

            # Catch all other server errors
            err = 'server_error'
            u = params['redirect_uri']
            return self._make_redirect_error_response(u, err)

```

### Rank 9 - `code_1028789` - `Provider._make_redirect_error_response`

- Repository: `NateFerrero/oauth2lib`

- Path: `oauth2lib/provider.py`

- Similarity: `0.620000`

- Source URL: https://github.com/NateFerrero/oauth2lib/blob/d161b010f8a596826050a09e5e94d59443cc12d9/oauth2lib/provider.py#L42-L59


```python

def _make_redirect_error_response(self, redirect_uri, err):
        """Return a HTTP 302 redirect response object containing the error.

        :param redirect_uri: Client redirect URI.
        :type redirect_uri: str
        :param err: OAuth error message.
        :type err: str
        :rtype: requests.Response
        """
        params = {
            'error': err,
            'response_type': None,
            'client_id': None,
            'redirect_uri': None
        }
        redirect = utils.build_url(redirect_uri, params)
        return self._make_response(headers={'Location': redirect},
                                   status_code=302)

```

### Rank 10 - `code_1028820` - `Provider._handle_exception`

- Repository: `NateFerrero/oauth2lib`

- Path: `oauth2lib/provider.py`

- Similarity: `0.615000`

- Source URL: https://github.com/NateFerrero/oauth2lib/blob/d161b010f8a596826050a09e5e94d59443cc12d9/oauth2lib/provider.py#L15-L22


```python

def _handle_exception(self, exc):
        """Handle an internal exception that was caught and suppressed.

        :param exc: Exception to process.
        :type exc: Exception
        """
        logger = logging.getLogger(__name__)
        logger.exception(exc)

```

### Rank 11 - `code_1032259` - `AuthorizationProvider.get_token_from_post_data`

- Repository: `NateFerrero/oauth2lib`

- Path: `oauth2lib/provider.py`

- Similarity: `0.610000`

- Source URL: https://github.com/NateFerrero/oauth2lib/blob/d161b010f8a596826050a09e5e94d59443cc12d9/oauth2lib/provider.py#L451-L482


```python

def get_token_from_post_data(self, data):
        """Get a token response from POST data.

        :param data: POST data containing authorization information.
        :type data: dict
        :rtype: requests.Response
        """
        try:
            # Verify OAuth 2.0 Parameters
            for x in ['grant_type', 'client_id', 'client_secret']:
                if not data.get(x):
                    raise TypeError("Missing required OAuth 2.0 POST param: {0}".format(x))
            
            # Handle get token from refresh_token
            if 'refresh_token' in data:
                return self.refresh_token(**data)

            # Handle get token from authorization code
            for x in ['redirect_uri', 'code']:
                if not data.get(x):
                    raise TypeError("Missing required OAuth 2.0 POST param: {0}".format(x))            
            return self.get_token(**data)
        except TypeError as exc:
            self._handle_exception(exc)

            # Catch missing parameters in request
            return self._make_json_error_response('invalid_request')
        except StandardError as exc:
            self._handle_exception(exc)

            # Catch all other server errors
            return self._make_json_error_response('server_error')

```

### Rank 12 - `code_1032272` - `Client.get_token`

- Repository: `NateFerrero/oauth2lib`

- Path: `oauth2lib/client.py`

- Similarity: `0.605000`

- Source URL: https://github.com/NateFerrero/oauth2lib/blob/d161b010f8a596826050a09e5e94d59443cc12d9/oauth2lib/client.py#L62-L80


```python

def get_token(self, code, **params):
        """Get an access token from the provider token URI.

        :param code: Authorization code.
        :type code: str
        :return: Dict containing access token, refresh token, etc.
        :rtype: dict
        """
        params['code'] = code
        if 'grant_type' not in params:
            params['grant_type'] = self.default_grant_type
        params.update({'client_id': self.client_id,
                       'client_secret': self.client_secret,
                       'redirect_uri': self.redirect_uri})
        response = self.http_post(self.token_uri, params)
        try:
            return response.json()
        except TypeError:
            return response.json

```

### Rank 13 - `code_1033461` - `AuthorizationProvider.get_authorization_code`

- Repository: `NateFerrero/oauth2lib`

- Path: `oauth2lib/provider.py`

- Similarity: `0.600000`

- Source URL: https://github.com/NateFerrero/oauth2lib/blob/d161b010f8a596826050a09e5e94d59443cc12d9/oauth2lib/provider.py#L205-L268


```python

def get_authorization_code(self,
                               response_type,
                               client_id,
                               redirect_uri,
                               **params):
        """Generate authorization code HTTP response.

        :param response_type: Desired response type. Must be exactly "code".
        :type response_type: str
        :param client_id: Client ID.
        :type client_id: str
        :param redirect_uri: Client redirect URI.
        :type redirect_uri: str
        :rtype: requests.Response
        """

        # Ensure proper response_type
        if response_type != 'code':
            err = 'unsupported_response_type'
            return self._make_redirect_error_response(redirect_uri, err)

        # Check redirect URI
        is_valid_redirect_uri = self.validate_redirect_uri(client_id,
                                                           redirect_uri)
        if not is_valid_redirect_uri:
            return self._invalid_redirect_uri_response()

        # Check conditions
        is_valid_client_id = self.validate_client_id(client_id)
        is_valid_access = self.validate_access()
        scope = params.get('scope', '')
        is_valid_scope = self.validate_scope(client_id, scope)

        # Return proper error responses on invalid conditions
        if not is_valid_client_id:
            err = 'unauthorized_client'
            return self._make_redirect_error_response(redirect_uri, err)

        if not is_valid_access:
            err = 'access_denied'
            return self._make_redirect_error_response(redirect_uri, err)

        if not is_valid_scope:
            err = 'invalid_scope'
            return self._make_redirect_error_response(redirect_uri, err)

        # Generate authorization code
        code = self.generate_authorization_code()

        # Save information to be used to validate later requests
        self.persist_authorization_code(client_id=client_id,
                                        code=code,
                                        scope=scope)

        # Return redirection response
        params.update({
            'code': code,
            'response_type': None,
            'client_id': None,
            'redirect_uri': None
        })
        redirect = utils.build_url(redirect_uri, params)
        return self._make_response(headers={'Location': redirect},
                                   status_code=302)

```

### Rank 14 - `code_1038344` - `AuthorizationProvider.refresh_token`

- Repository: `NateFerrero/oauth2lib`

- Path: `oauth2lib/provider.py`

- Similarity: `0.595000`

- Source URL: https://github.com/NateFerrero/oauth2lib/blob/d161b010f8a596826050a09e5e94d59443cc12d9/oauth2lib/provider.py#L270-L336


```python

def refresh_token(self,
                      grant_type,
                      client_id,
                      client_secret,
                      refresh_token,
                      **params):
        """Generate access token HTTP response from a refresh token.

        :param grant_type: Desired grant type. Must be "refresh_token".
        :type grant_type: str
        :param client_id: Client ID.
        :type client_id: str
        :param client_secret: Client secret.
        :type client_secret: str
        :param refresh_token: Refresh token.
        :type refresh_token: str
        :rtype: requests.Response
        """

        # Ensure proper grant_type
        if grant_type != 'refresh_token':
            return self._make_json_error_response('unsupported_grant_type')

        # Check conditions
        is_valid_client_id = self.validate_client_id(client_id)
        is_valid_client_secret = self.validate_client_secret(client_id,
                                                             client_secret)
        scope = params.get('scope', '')
        is_valid_scope = self.validate_scope(client_id, scope)
        data = self.from_refresh_token(client_id, refresh_token, scope)
        is_valid_refresh_token = data is not None

        # Return proper error responses on invalid conditions
        if not (is_valid_client_id and is_valid_client_secret):
            return self._make_json_error_response('invalid_client')

        if not is_valid_scope:
            return self._make_json_error_response('invalid_scope')

        if not is_valid_refresh_token:
            return self._make_json_error_response('invalid_grant')

        # Discard original refresh token
        self.discard_refresh_token(client_id, refresh_token)

        # Generate access tokens once all conditions have been met
        access_token = self.generate_access_token()
        token_type = self.token_type
        expires_in = self.token_expires_in
        refresh_token = self.generate_refresh_token()

        # Save information to be used to validate later requests
        self.persist_token_information(client_id=client_id,
                                       scope=scope,
                                       access_token=access_token,
                                       token_type=token_type,
                                       expires_in=expires_in,
                                       refresh_token=refresh_token,
                                       data=data)

        # Return json response
        return self._make_json_response({
            'access_token': access_token,
            'token_type': token_type,
            'expires_in': expires_in,
            'refresh_token': refresh_token
        })

```

### Rank 15 - `code_1007230` - `build_url`

- Repository: `NateFerrero/oauth2lib`

- Path: `oauth2lib/utils.py`

- Similarity: `0.590000`

- Source URL: https://github.com/NateFerrero/oauth2lib/blob/d161b010f8a596826050a09e5e94d59443cc12d9/oauth2lib/utils.py#L41-L65


```python

def build_url(base, additional_params=None):
    """Construct a URL based off of base containing all parameters in
    the query portion of base plus any additional parameters.

    :param base: Base URL
    :type base: str
    ::param additional_params: Additional query parameters to include.
    :type additional_params: dict
    :rtype: str
    """
    url = urlparse.urlparse(base)
    query_params = {}
    query_params.update(urlparse.parse_qsl(url.query, True))
    if additional_params is not None:
        query_params.update(additional_params)
        for k, v in additional_params.iteritems():
            if v is None:
                query_params.pop(k)

    return urlparse.urlunparse((url.scheme,
                                url.netloc,
                                url.path,
                                url.params,
                                urllib.urlencode(query_params),
                                url.fragment))

```

### Rank 16 - `code_1042338` - `ResourceProvider.get_authorization`

- Repository: `NateFerrero/oauth2lib`

- Path: `oauth2lib/provider.py`

- Similarity: `0.590000`

- Source URL: https://github.com/NateFerrero/oauth2lib/blob/d161b010f8a596826050a09e5e94d59443cc12d9/oauth2lib/provider.py#L573-L586


```python

def get_authorization(self):
        """Get authorization object representing status of authentication."""
        auth = self.authorization_class()
        header = self.get_authorization_header()
        if not header or not header.split:
            return auth
        header = header.split()
        if len(header) > 1 and header[0] == 'Bearer':
            auth.is_oauth = True
            access_token = header[1]
            self.validate_access_token(access_token, auth)
            if not auth.is_valid:
                auth.error = 'access_denied'
        return auth

```

### Rank 17 - `code_1018744` - `_get_uri_from_request`

- Repository: `lepture/flask-oauthlib`

- Path: `flask_oauthlib/utils.py`

- Similarity: `0.585000`

- Source URL: https://github.com/lepture/flask-oauthlib/blob/9e6f152a5bb360e7496210da21561c3e6d41b0e1/flask_oauthlib/utils.py#L8-L17


```python

def _get_uri_from_request(request):
    """
    The uri returned from request.uri is not properly urlencoded
    (sometimes it's partially urldecoded) This is a weird hack to get
    werkzeug to return the proper urlencoded string uri
    """
    uri = request.base_url
    if request.query_string:
        uri += '?' + request.query_string.decode('utf-8')
    return uri

```

### Rank 18 - `code_1042916` - `urlunparse`

- Repository: `google/grumpy`

- Path: `third_party/stdlib/urlparse.py`

- Similarity: `0.580000`

- Source URL: https://github.com/google/grumpy/blob/3ec87959189cfcdeae82eb68a47648ac25ceb10b/third_party/stdlib/urlparse.py#L345-L353


```python

def urlunparse(data):
    """Put a parsed URL back together again.  This may result in a
    slightly different, but equivalent URL, if the URL that was parsed
    originally had redundant delimiters, e.g. a ? with an empty query
    (the draft states that these are equivalent)."""
    scheme, netloc, url, params, query, fragment = data
    if params:
        url = "%s;%s" % (url, params)
    return urlunsplit((scheme, netloc, url, query, fragment))

```

### Rank 19 - `code_1025484` - `GChart.fromurl`

- Repository: `appknox/google-chartwrapper`

- Path: `GChartWrapper/GChart.py`

- Similarity: `0.575000`

- Source URL: https://github.com/appknox/google-chartwrapper/blob/3769aecbef6c83b6cd93ee72ece478ffe433ac57/GChartWrapper/GChart.py#L203-L214


```python

def fromurl(cls, qs):
        """
        Reverse a chart URL or dict into a GChart instance
        
        >>> G = GChart.fromurl('http://chart.apis.google.com/chart?...')
        >>> G
        <GChartWrapper.GChart instance at...>
        >>> G.image().save('chart.jpg','JPEG')
        """
        if isinstance(qs, dict):
            return cls(**qs)
        return cls(**dict(parse_qsl(qs[qs.index('?')+1:])))

```

### Rank 20 - `code_1016826` - `CopyDoc._parse_href`

- Repository: `nprapps/copydoc`

- Path: `copydoc.py`

- Similarity: `0.570000`

- Source URL: https://github.com/nprapps/copydoc/blob/e1ab09b287beb0439748c319cf165cbc06c66624/copydoc.py#L189-L195


```python

def _parse_href(self, href):
        """
        Extract "real" URL from Google redirected url by getting `q`
        querystring parameter.
        """
        params = parse_qs(urlsplit(href).query)
        return params.get('q')

```

## Generation Tests

The following behavior-oriented tests are the current generation evaluation cases. `csn_11772` is intentionally omitted.

### csn_8884 tests

Source: `backend/app/generation_tests/csn_8884.py`


```python

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


TEST_CASES = [
    test_removes_dead_code_from_try_body,
    test_removes_dead_code_from_all_try_statement_regions,
    test_preserves_try_source_location,
    test_preserves_try_handlers_and_reachable_structure,
]

```

### csn_3846 tests

Source: `backend/app/generation_tests/csn_3846.py`


```python

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


TEST_CASES = [
    test_identifies_same_name_setter,
    test_identifies_same_name_deleter,
    test_rejects_different_name_setter,
    test_rejects_simple_decorator,
    test_rejects_method_without_decorator,
]

```

### csn_42 tests

Source: `backend/app/generation_tests/csn_42.py`


```python

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

```

### csn_9388 tests

Source: `backend/app/generation_tests/csn_9388.py`


```python

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

```
