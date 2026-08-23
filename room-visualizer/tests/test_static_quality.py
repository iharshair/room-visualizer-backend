"""Static quality gates over the delivered tree (Requirements 2.8, 9.7, 10.8, 11.1, 11.6, 11.7, 11.8).

Every other test module drives code. This one reads it. The guarantees here are
properties of the *files as delivered* -- a widget that cannot leak style into a
host page, a source tree with nothing standing in for missing behaviour, a
dependency manifest that reproduces one environment, a README that documents what
an operator has to know, and the file layout Requirement 11.1 fixes by name.
None of those can be observed by calling a function, and all of them regress
silently, which is why they are asserted rather than reviewed.

**Property 34** parses `frontend/css/visualizer.css` and holds every selector to
the scoping invariant the stylesheet's own header states: each selector begins
with `.rv-root`, no compound selector names a bare element type, and nothing
targets `:root`, `:host`, or the universal selector. The parser is hand-rolled
because the pinned test dependencies are pytest, pytest-cov, hypothesis, and
psutil -- there is no CSS library to lean on. It is nonetheless written to be
at-rule aware: the current stylesheet deliberately contains no at-rules at all
(so that a percentage keyframe selector like `0%` can never be mistaken for an
unscoped selector), but a future `@media` block must neither slip past the check
nor invent a failure. `_parse_stylesheet` therefore recurses into conditional
group rules and skips the bodies of `@keyframes` and friends, and
`test_the_parser_recurses_into_conditional_group_at_rules` proves both halves.

**Property 35** scans every delivered Python and JavaScript source for placeholder
markers, and additionally walks the Python with `ast` looking for function bodies
that consist solely of `pass`, `...`, or a raised `NotImplementedError`. Two
kinds of declaration are exempt, because in both the empty body *is* the content:
`@abstractmethod` members of the Segmenter interface, and members of a
`typing.Protocol`. The exemption is not open-ended --
`test_the_only_stub_bodies_are_declared_interface_members` pins the exempt set to
an exact inventory, so a new empty body cannot arrive wearing a decorator and go
unnoticed. JavaScript is checked by marker only: `() => {}` is a legitimate
no-op rejection handler in `visualizer.js`, and there is no JS parser available to
tell that apart from an unwritten function.

A static check that finds nothing is indistinguishable from a static check that
cannot find anything, so every detector here is paired with a test that feeds it a
synthetic violation and asserts it is reported. Those guards are the reason the
passing results above mean something.

**Marker spelling.** This module is itself a delivered source under `tests/`, so
it is one of the files its own marker scan reads. Spelling any marker literally
would make the scan fail on this file. Every marker is therefore assembled at
runtime from fragments in `_MARKER_FRAGMENTS`; the joined form exists only in
memory, never in this file's bytes.

**Validates: Requirements 2.8, 9.7, 10.8, 11.1, 11.6, 11.7, 11.8**
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pytest
from hypothesis import given, settings, strategies as st

from backend.config import Settings

# --------------------------------------------------------------------------- #
# The delivered tree
# --------------------------------------------------------------------------- #
#
# "Delivered" is defined here rather than left implicit, because Property 35
# quantifies over it. It is everything a clone receives and runs:
#
#   * Python under `backend/` (the service), `scripts/` (the Setup_Tool), and
#     `tests/` (the Test_Suite -- Requirement 13 makes the suite a deliverable,
#     so a placeholder inside a test is as much a defect as one in the service).
#   * JavaScript under `frontend/js/`.
#   * The stylesheet and the demo host page, which are delivered sources too and
#     are covered by the marker half of the scan.
#
# Excluded: anything under a dot-directory (`.venv`, `.hypothesis`,
# `.pytest_cache`) and `__pycache__`. Those are installed or generated, not
# delivered.

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

_CSS_PATH = _PROJECT_ROOT / "frontend" / "css" / "visualizer.css"
_README_PATH = _PROJECT_ROOT / "README.md"
_REQUIREMENTS_PATH = _PROJECT_ROOT / "backend" / "requirements.txt"

_PYTHON_SOURCE_DIRS = ("backend", "scripts", "tests")
_JS_SOURCE_DIRS = ("frontend/js",)
_OTHER_DELIVERED_SOURCES = ("frontend/css/visualizer.css", "frontend/index.html")

_EXCLUDED_DIR_NAMES = frozenset({"__pycache__", "node_modules"})


def _is_delivered(path: Path) -> bool:
    """False for generated or installed paths that merely sit in the tree."""
    parts = path.relative_to(_PROJECT_ROOT).parts
    return not any(part.startswith(".") or part in _EXCLUDED_DIR_NAMES for part in parts)


def _iter_sources(relative_dirs: Sequence[str], suffix: str) -> Iterator[Path]:
    for relative in relative_dirs:
        base = _PROJECT_ROOT / relative
        if not base.is_dir():  # pragma: no cover - guarded by the inventory test
            continue
        for path in sorted(base.rglob(f"*{suffix}")):
            if path.is_file() and _is_delivered(path):
                yield path


def _python_sources() -> tuple[Path, ...]:
    return tuple(_iter_sources(_PYTHON_SOURCE_DIRS, ".py"))


def _text_sources() -> tuple[Path, ...]:
    """Every delivered source the marker scan reads."""
    others = tuple(_PROJECT_ROOT / relative for relative in _OTHER_DELIVERED_SOURCES)
    return _python_sources() + tuple(_iter_sources(_JS_SOURCE_DIRS, ".js")) + others


PYTHON_SOURCES: tuple[Path, ...] = _python_sources()
TEXT_SOURCES: tuple[Path, ...] = _text_sources()


def _relative(path: Path) -> str:
    return path.relative_to(_PROJECT_ROOT).as_posix()


# --------------------------------------------------------------------------- #
# CSS parsing
# --------------------------------------------------------------------------- #
#
# A stylesheet is scanned rather than regex-matched, because the two things that
# break a regex are both present in this file: a header comment containing
# `#my-container { --rv-accent: #2f6f4f; }` (braces inside a comment) and
# declarations containing quoted strings (`"Segoe UI"`). Comments are removed
# first, with newlines preserved so reported line numbers stay true, and the
# scanner tracks quote state so a brace or semicolon inside a string is inert.

#: At-rules whose body contains further rules, and which are therefore recursed
#: into. A selector inside `@media` is as much a selector as one at top level.
_CONDITIONAL_GROUP_AT_RULES = frozenset(
    {"media", "supports", "layer", "container", "scope", "document"}
)

#: At-rules whose body is *not* a list of style rules. `@keyframes` blocks are
#: keyed by percentages and `from`/`to`, `@font-face` and `@property` hold plain
#: declarations. Descending into these is what would make `0%` look like an
#: unscoped element selector, so their bodies are skipped.
_NON_SELECTOR_AT_RULES = frozenset(
    {
        "keyframes",
        "-webkit-keyframes",
        "-moz-keyframes",
        "font-face",
        "font-feature-values",
        "counter-style",
        "page",
        "property",
        "viewport",
        "color-profile",
    }
)

#: Pseudo-classes whose arguments are themselves selectors, and so are validated
#: recursively -- `.rv-root :is(button)` must fail like `.rv-root button` does.
#: `:nth-child(2n+1)` and `:lang(en)` are deliberately absent: their arguments
#: are not selectors.
_SELECTOR_ARG_PSEUDOS = frozenset(
    {"is", "where", "not", "has", "matches", "any", "-moz-any", "-webkit-any"}
)

#: Pseudo-classes and pseudo-elements that reach outside the component subtree
#: however they are qualified.
_GLOBAL_PSEUDOS = ("root", "host", "host-context", "scope")

#: The component root class. Requirement 10.8 in one token.
_ROOT_CLASS = ".rv-root"

#: A compound selector may begin only with one of these: a class, an id, an
#: attribute test, a pseudo, or the nesting selector. Anything else is an element
#: type name (`button`, `html`, `body`) or the universal selector.
_COMPOUND_OPENERS = ".#[:&"

_ROOT_PREFIX_RE = re.compile(r"\.rv-root(?![\w-])")
_AT_NAME_RE = re.compile(r"@(-?[\w-]+)")


def _strip_css_comments(css: str) -> str:
    """Replace every `/* ... */` with equivalent whitespace.

    Length and newlines are preserved so that offsets computed on the result
    still map to the original file's line numbers.
    """
    out: list[str] = []
    index, length = 0, len(css)
    quote: str | None = None
    while index < length:
        char = css[index]
        if quote is not None:
            out.append(char)
            if char == "\\" and index + 1 < length:
                out.append(css[index + 1])
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char in "\"'":
            quote = char
            out.append(char)
            index += 1
            continue
        if char == "/" and css.startswith("/*", index):
            close = css.find("*/", index + 2)
            end = length if close == -1 else close + 2
            out.extend("\n" if inner == "\n" else " " for inner in css[index:end])
            index = end
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _matching_brace(text: str, open_index: int, end: int) -> int:
    """Index of the `}` closing the `{` at ``open_index``, or ``end`` if unclosed."""
    depth = 0
    index = open_index
    quote: str | None = None
    while index < end:
        char = text[index]
        if quote is not None:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char in "\"'":
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return end


@dataclass(frozen=True, slots=True)
class _Block:
    """One `prelude { body }` rule, or one `prelude ;` statement."""

    prelude: str
    prelude_index: int
    body_start: int | None
    body_end: int | None


def _iter_blocks(text: str, start: int, end: int) -> Iterator[_Block]:
    """Yield the top-level blocks and statements between ``start`` and ``end``."""
    index = start
    prelude_index = start
    quote: str | None = None
    while index < end:
        char = text[index]
        if quote is not None:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char in "\"'":
            quote = char
            index += 1
            continue
        if char == "{":
            close = _matching_brace(text, index, end)
            yield _Block(text[prelude_index:index], prelude_index, index + 1, close)
            index = close + 1
            prelude_index = index
            continue
        if char == ";":
            prelude = text[prelude_index:index]
            if prelude.strip():
                yield _Block(prelude, prelude_index, None, None)
            index += 1
            prelude_index = index
            continue
        index += 1


def _split_top_level(text: str, separators: str) -> list[tuple[str, int]]:
    """Split on ``separators`` outside parentheses, brackets, and strings.

    Returns each piece with its offset in ``text``, so a violation can be
    reported against the line it is on.
    """
    pieces: list[tuple[str, int]] = []
    current: list[str] = []
    piece_start = 0
    parens = brackets = 0
    quote: str | None = None
    for offset, char in enumerate(text):
        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
            current.append(char)
            continue
        if char == "(":
            parens += 1
        elif char == ")":
            parens = max(0, parens - 1)
        elif char == "[":
            brackets += 1
        elif char == "]":
            brackets = max(0, brackets - 1)
        if parens == 0 and brackets == 0 and char in separators:
            pieces.append(("".join(current), piece_start))
            current = []
            piece_start = offset + 1
            continue
        if not current:
            piece_start = offset
        current.append(char)
    pieces.append(("".join(current), piece_start))
    return pieces


def _split_compounds(selector: str) -> list[str]:
    """Split a selector into compound selectors on combinators and whitespace."""
    return [
        piece.strip()
        for piece, _ in _split_top_level(selector, " \t\n\r\f>+~")
        if piece.strip()
    ]


@dataclass(frozen=True, slots=True)
class CssSelector:
    """One selector from one comma-separated list, with where it came from."""

    text: str
    line: int
    at_context: tuple[str, ...]
    nested: bool

    def __str__(self) -> str:  # pragma: no cover - assertion messages only
        where = f"line {self.line}"
        if self.at_context:
            where += " in @" + " @".join(self.at_context)
        return f"{self.text!r} ({where})"


def _line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _parse_stylesheet(css: str) -> tuple[tuple[CssSelector, ...], tuple[str, ...]]:
    """Return every style-rule selector in ``css``, plus the at-rules seen.

    At-rules are returned as well as recursed through, so a test can assert what
    the stylesheet actually contains instead of assuming.
    """
    text = _strip_css_comments(css)
    selectors: list[CssSelector] = []
    at_rules: list[str] = []

    def walk(start: int, end: int, at_context: tuple[str, ...], nested: bool) -> None:
        for block in _iter_blocks(text, start, end):
            prelude = block.prelude.strip()
            if not prelude:
                continue
            if prelude.startswith("@"):
                match = _AT_NAME_RE.match(prelude)
                name = match.group(1).lower() if match else ""
                at_rules.append(name)
                if block.body_start is None or block.body_end is None:
                    continue
                if name in _CONDITIONAL_GROUP_AT_RULES:
                    walk(block.body_start, block.body_end, at_context + (name,), nested)
                # Any other at-rule body holds keyframe stops or declarations,
                # not selectors, so it is deliberately not descended into.
                continue
            if block.body_start is None or block.body_end is None:
                # A declaration, reached while walking a rule body. Not a rule.
                continue
            for piece, offset in _split_top_level(block.prelude, ","):
                if piece.strip():
                    selectors.append(
                        CssSelector(
                            text=piece.strip(),
                            line=_line_of(text, block.prelude_index + offset),
                            at_context=at_context,
                            nested=nested,
                        )
                    )
            # Descend, so nested rules (CSS nesting) are checked rather than
            # silently ignored. Declarations inside are skipped by the branch
            # above; a nested rule arrives with `nested=True`.
            walk(block.body_start, block.body_end, at_context, True)

    walk(0, len(text), (), False)
    return tuple(selectors), tuple(at_rules)


def _functional_pseudo_args(compound: str) -> Iterator[str]:
    """Yield the selector arguments of every selector-taking pseudo in ``compound``."""
    for match in re.finditer(r":(-?[\w-]+)\(", compound):
        if match.group(1).lower() not in _SELECTOR_ARG_PSEUDOS:
            continue
        depth = 0
        for index in range(match.end() - 1, len(compound)):
            if compound[index] == "(":
                depth += 1
            elif compound[index] == ")":
                depth -= 1
                if depth == 0:
                    inner = compound[match.end() : index]
                    for piece, _ in _split_top_level(inner, ","):
                        if piece.strip():
                            yield piece.strip()
                    break


def _compound_violations(compound: str) -> list[str]:
    """Scoping violations inside one compound selector."""
    violations: list[str] = []
    body = compound[1:] if compound.startswith("&") else compound
    if not body:
        return violations

    # The universal selector, but not the `*=` of an attribute substring test,
    # which `_split_top_level` leaves inside its brackets.
    outside_brackets = re.sub(r"\[[^\]]*\]", "", body)
    outside_brackets = re.sub(r"\((?:[^()]|\([^()]*\))*\)", "", outside_brackets)
    if "*" in outside_brackets:
        violations.append(f"universal selector in {compound!r}")

    if body[0] not in _COMPOUND_OPENERS:
        violations.append(f"bare element type selector {compound!r}")

    for pseudo in _GLOBAL_PSEUDOS:
        if re.search(rf"::?{re.escape(pseudo)}(?![\w-])", body):
            violations.append(f"global pseudo :{pseudo} in {compound!r}")

    for argument in _functional_pseudo_args(body):
        for inner in _split_compounds(argument):
            violations.extend(
                f"{message} (inside {compound!r})" for message in _compound_violations(inner)
            )
    return violations


def selector_violations(selector: CssSelector) -> list[str]:
    """Every way ``selector`` breaks the Property 34 scoping invariant.

    Empty means compliant. The list is returned rather than asserted so the
    caller can report all of them at once and so the detector itself is testable.
    """
    compounds = _split_compounds(selector.text)
    if not compounds:
        return ["empty selector"]

    violations: list[str] = []
    first = compounds[0]
    if selector.nested:
        # A nested rule is already inside a scoped ancestor; it has to either
        # reference it explicitly or re-state the root class.
        if not first.startswith("&") and not _ROOT_PREFIX_RE.match(first):
            violations.append(
                f"nested selector {selector.text!r} neither starts with '&' nor with {_ROOT_CLASS}"
            )
    elif not _ROOT_PREFIX_RE.match(first):
        violations.append(f"selector {selector.text!r} is not scoped under {_ROOT_CLASS}")

    for compound in compounds:
        violations.extend(_compound_violations(compound))
    return violations


def _selectors_of(css: str) -> tuple[CssSelector, ...]:
    return _parse_stylesheet(css)[0]


CSS_TEXT: str = _CSS_PATH.read_text(encoding="utf-8")
CSS_SELECTORS: tuple[CssSelector, ...] = _selectors_of(CSS_TEXT)


# --------------------------------------------------------------------------- #
# Property 34 -- component CSS is scoped to the container (Requirement 10.8)
# --------------------------------------------------------------------------- #
#
# The quantifier is over a finite set: the selectors this stylesheet contains.
# Hypothesis is used anyway, with `sampled_from`, because it exhausts a pool that
# small -- 100 examples over roughly 60 selectors visits every one, and does so
# with shrinking, so a failure is reported against a single named selector rather
# than as a wall of them. The exhaustive loop is asserted separately below, so
# neither form depends on the other for coverage.

_PROPERTY_SETTINGS = settings(max_examples=100, deadline=None)


# Feature: ai-room-tile-visualizer, Property 34: All component CSS is scoped to
# the container
@_PROPERTY_SETTINGS
@given(selector=st.sampled_from(CSS_SELECTORS))
def test_property_34_every_css_selector_is_scoped_to_the_container(selector):
    """For any selector in `frontend/css/visualizer.css`, the selector is scoped
    under the component root class and no selector targets a bare element type, a
    global pseudo-class, or `:root`.

    This is what makes the widget safe to drop into a page whose styles nobody
    controls. A single `button { ... }` rule in a shipped stylesheet restyles the
    host's buttons, and the failure shows up on the customer's page rather than
    here, which is why it is pinned mechanically.

    **Validates: Requirements 10.8**
    """
    violations = selector_violations(selector)
    assert not violations, f"{selector}: " + "; ".join(violations)


def test_the_whole_stylesheet_is_scoped_not_merely_a_sample():
    """Property 34 over the entire selector set at once.

    `sampled_from` with 100 examples exhausts this pool, but that is a property
    of the pool's current size rather than a guarantee. This loop does not
    depend on it, and reports every offender in one message instead of shrinking
    to one.
    """
    offenders = {
        str(selector): selector_violations(selector)
        for selector in CSS_SELECTORS
        if selector_violations(selector)
    }
    assert not offenders, f"unscoped selectors in {_relative(_CSS_PATH)}: {offenders}"


def test_the_stylesheet_parses_to_a_plausible_selector_inventory():
    """Guard for Property 34: a parser that returned nothing would pass it.

    Pins the shape of what was parsed -- a substantial number of selectors, every
    one mentioning the component prefix, and the root class itself present -- so
    a scanner that silently stopped at the first comment or brace is caught.
    """
    assert len(CSS_SELECTORS) > 40, (
        f"only {len(CSS_SELECTORS)} selectors parsed from {_relative(_CSS_PATH)}; "
        "the scanner is probably dropping rules"
    )
    assert any(selector.text == _ROOT_CLASS for selector in CSS_SELECTORS)
    assert all("rv-" in selector.text for selector in CSS_SELECTORS)

    # Every class token the stylesheet uses is component-namespaced. The scoping
    # check above already forbids reaching outside `.rv-root`; this pins the
    # naming convention that keeps the classes themselves collision-free.
    for selector in CSS_SELECTORS:
        for token in re.findall(r"\.(-?[\w-]+)", selector.text):
            assert token.startswith("rv-"), f"{selector}: class .{token} is not rv-prefixed"


def test_the_stylesheet_declares_its_theming_tokens_on_the_root():
    """Requirement 10.8's positive half: the documented theming contract.

    The tokens the design names are declared on `.rv-root`, so a host page can
    retheme the widget by setting them on its own container, and nowhere else --
    a `:root` declaration would be both a leak and a token a host cannot override
    locally.
    """
    root_rule = re.search(
        r"^\.rv-root\s*\{(.*?)^\}", _strip_css_comments(CSS_TEXT), re.S | re.M
    )
    assert root_rule is not None, ".rv-root rule not found"
    for token in ("--rv-accent", "--rv-radius", "--rv-gap"):
        assert token in root_rule.group(1), f"{token} is not declared on {_ROOT_CLASS}"


# --------------------------------------------------------------------------- #
# CSS parser guards
# --------------------------------------------------------------------------- #
#
# The detector has to be shown to fire. Each case below is a violation the
# stylesheet does not currently contain, so without these the passing result
# above could equally mean "compliant" or "blind".


@pytest.mark.parametrize(
    ("css", "expected"),
    [
        ("button { color: red; }", "not scoped"),
        ("html, body { margin: 0; }", "not scoped"),
        (":root { --rv-accent: red; }", "not scoped"),
        ("* { box-sizing: border-box; }", "not scoped"),
        (".rv-root button { font: inherit; }", "bare element type"),
        (".rv-root * { margin: 0; }", "universal selector"),
        (".rv-root :is(button, a) { color: red; }", "bare element type"),
        (".rv-root:root { color: red; }", "global pseudo :root"),
        (".rv-root .rv-a, canvas { display: block; }", "not scoped"),
        (".rv-rootish .rv-a { display: block; }", "not scoped"),
    ],
    ids=[
        "bare_element_at_top_level",
        "html_and_body",
        "root_pseudo_class",
        "universal_at_top_level",
        "bare_element_descendant",
        "universal_descendant",
        "bare_element_inside_is",
        "root_pseudo_on_the_component_root",
        "one_bad_selector_in_a_list",
        "root_class_prefix_is_not_a_substring_match",
    ],
)
def test_the_scoping_check_reports_known_violations(css, expected):
    """Each synthetic stylesheet must be reported, and reported for the right
    reason -- `.rv-rootish` in particular must not pass as `.rv-root`."""
    messages = [
        message
        for selector in _selectors_of(css)
        for message in selector_violations(selector)
    ]
    assert any(expected in message for message in messages), (
        f"{css!r} was not reported for {expected!r}; got {messages}"
    )


@pytest.mark.parametrize(
    "css",
    [
        ".rv-root { display: flex; }",
        ".rv-root .rv-a { display: block; }",
        ".rv-root .rv-a > .rv-b + .rv-c ~ .rv-d { color: red; }",
        '.rv-root[aria-busy="true"] .rv-a { opacity: 0.5; }',
        '.rv-root .rv-a:focus-visible, .rv-root .rv-b[data-x="1"] { outline: 1px solid; }',
        '.rv-root .rv-a[class*="rv-b"] { color: red; }',
        ".rv-root .rv-a:not(.rv-b):nth-child(2n+1) { color: red; }",
        ".rv-root .rv-a:has(.rv-b) { color: red; }",
        '.rv-root .rv-a { content: "}"; }',
        '.rv-root .rv-a { font-family: "Segoe UI", sans-serif; }',
    ],
    ids=[
        "root_alone",
        "descendant",
        "all_combinators",
        "attribute_on_the_root",
        "selector_list_with_pseudo_and_attribute",
        "attribute_substring_match_is_not_the_universal_selector",
        "functional_pseudo_with_non_selector_arguments",
        "has_with_a_scoped_argument",
        "a_brace_inside_a_string_value",
        "a_comma_inside_a_string_value",
    ],
)
def test_the_scoping_check_accepts_compliant_selectors(css):
    """The complement: a check that rejected everything would also pass the
    violation cases above, so the accepting half is pinned too."""
    for selector in _selectors_of(css):
        assert not selector_violations(selector), f"{selector} was wrongly reported"


def test_comments_cannot_be_mistaken_for_rules():
    """The real stylesheet's header comment contains
    `#my-container { --rv-accent: #2f6f4f; }` as documentation. A scanner that
    did not strip comments first would report it as an unscoped id selector."""
    css = """
    /* Theme it with `#my-container { --rv-accent: #2f6f4f; }` -- and note
       this comment contains braces, a semicolon, and an apostrophe. */
    .rv-root .rv-a { color: red; }
    """
    assert [selector.text for selector in _selectors_of(css)] == [".rv-root .rv-a"]


def test_the_parser_recurses_into_conditional_group_at_rules():
    """The stylesheet ships no at-rules today, deliberately. The parser is still
    at-rule aware in both directions, and this is where that is proven.

    A selector inside `@media` must be checked -- otherwise adding one media
    query would open a hole big enough to hide an unscoped rule in. A keyframe
    stop like `0%` must *not* be checked, because it is not a selector at all and
    reporting it would be a false failure that pressures the next author into
    weakening the check.
    """
    scoped_media = "@media (min-width: 30rem) { .rv-root .rv-a { display: flex; } }"
    assert [s.text for s in _selectors_of(scoped_media)] == [".rv-root .rv-a"]
    assert _parse_stylesheet(scoped_media)[1] == ("media",)
    assert all(not selector_violations(s) for s in _selectors_of(scoped_media))

    unscoped_media = "@media (min-width: 30rem) { button { display: flex; } }"
    messages = [m for s in _selectors_of(unscoped_media) for m in selector_violations(s)]
    assert any("not scoped" in message for message in messages), messages

    keyframes = "@keyframes rv-spin { 0% { opacity: 0; } to { opacity: 1; } }"
    assert _selectors_of(keyframes) == ()
    assert _parse_stylesheet(keyframes)[1] == ("keyframes",)

    # `@font-face` and a statement at-rule must not produce selectors either.
    assert _selectors_of("@font-face { font-family: rv; src: url(a.woff2); }") == ()
    assert _selectors_of('@import "other.css";') == ()
    assert _parse_stylesheet('@import "other.css";')[1] == ("import",)


def test_the_parser_checks_nested_rules_rather_than_ignoring_them():
    """CSS nesting is not used in the stylesheet. If it ever is, nested selectors
    must be validated -- and `&`-relative ones must be accepted, since their
    scope comes from the enclosing rule."""
    nested_ok = ".rv-root { color: red; &:hover { color: blue; } .rv-a { color: green; } }"
    nested = {s.text: s for s in _selectors_of(nested_ok)}
    assert set(nested) == {".rv-root", "&:hover", ".rv-a"}
    assert nested["&:hover"].nested and not nested[".rv-root"].nested
    assert not selector_violations(nested["&:hover"])
    # A nested rule that neither references the parent nor re-states the root is
    # unanchored, and a nested bare element type is a leak either way.
    assert selector_violations(nested[".rv-a"])
    bad = ".rv-root { & button { color: red; } }"
    messages = [m for s in _selectors_of(bad) for m in selector_violations(s)]
    assert any("bare element type" in message for message in messages), messages


# --------------------------------------------------------------------------- #
# Placeholder markers
# --------------------------------------------------------------------------- #
#
# Assembled at runtime from fragments: see the module docstring. The joined
# strings never appear in this file's bytes, so the scan can read this file like
# any other delivered source.

_MARKER_FRAGMENTS: tuple[tuple[str, ...], ...] = (
    ("TO", "DO"),
    ("FIX", "ME"),
    ("XX", "X"),
    ("HAC", "K"),
    ("TB", "D"),
    ("WI", "P"),
    ("unimplement", "ed"),
    ("not ", "implemented"),
    ("placeholder ", "implementation"),
    ("coming ", "soon"),
    ("left as an ", "exercise"),
)

MARKERS: tuple[str, ...] = tuple("".join(fragments) for fragments in _MARKER_FRAGMENTS)


def _marker_pattern() -> re.Pattern[str]:
    """Case-insensitive, word-bounded alternation over every marker.

    Word boundaries matter in both directions. Without them the short markers
    match inside ordinary words -- and a check that fires on prose is a check
    somebody switches off. The gap between the words of a multi-word marker
    admits comment punctuation as well as whitespace, so one wrapped across two
    comment lines is still found.
    """
    gap = r"[\s]*(?:[#*/]+[ \t]*)?[\s]*"
    alternatives = []
    for marker in MARKERS:
        escaped = gap.join(re.escape(word) for word in marker.split())
        alternatives.append(rf"\b{escaped}\b")
    return re.compile("|".join(alternatives), re.IGNORECASE)


MARKER_RE: re.Pattern[str] = _marker_pattern()


def marker_hits(text: str) -> list[str]:
    """`line: matched text` for every placeholder marker in ``text``.

    Matched over the whole text rather than line by line, so a marker split
    across a wrapped comment is caught; the offset is mapped back to a line
    number for the report.
    """
    return [
        f"{text.count(chr(10), 0, match.start()) + 1}: {match.group(0).strip()!r}"
        for match in MARKER_RE.finditer(text)
    ]


# --------------------------------------------------------------------------- #
# Placeholder function bodies
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class StubBody:
    """A function whose body is only `pass`, `...`, or a raised NotImplementedError."""

    path: str
    qualname: str
    line: int
    form: str
    exempt_as: str | None

    def __str__(self) -> str:  # pragma: no cover - assertion messages only
        return f"{self.path}:{self.line} {self.qualname} ({self.form})"


def _is_abstract(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    names = {
        decorator.attr if isinstance(decorator, ast.Attribute) else getattr(decorator, "id", "")
        for decorator in node.decorator_list
    }
    return "abstractmethod" in names or "abstractproperty" in names


def _is_protocol(node: ast.ClassDef | None) -> bool:
    if node is None:
        return False
    for base in node.bases:
        name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
        if name == "Protocol":
            return True
        # `Protocol[T]`
        if isinstance(base, ast.Subscript):
            inner = base.value
            inner_name = inner.attr if isinstance(inner, ast.Attribute) else getattr(inner, "id", "")
            if inner_name == "Protocol":
                return True
    return False


def _stub_form(body: list[ast.stmt]) -> str | None:
    """The placeholder form of ``body``, ignoring a leading docstring, or None."""
    statements = [
        statement
        for statement in body
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )
    ]
    if len(statements) != 1:
        return None
    statement = statements[0]
    if isinstance(statement, ast.Pass):
        return "pass"
    if (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and statement.value.value is Ellipsis
    ):
        return "ellipsis"
    if isinstance(statement, ast.Raise):
        raised = statement.exc
        if isinstance(raised, ast.Call):
            raised = raised.func
        if isinstance(raised, ast.Name) and raised.id == "NotImplementedError":
            return "raise NotImplementedError"
        if isinstance(raised, ast.Attribute) and raised.attr == "NotImplementedError":
            return "raise NotImplementedError"
    return None


def stub_bodies(source: str, label: str) -> list[StubBody]:
    """Every placeholder-bodied function in ``source``, exempt ones included.

    Exempt declarations are returned rather than filtered out so that a test can
    pin the exemption inventory instead of trusting it.
    """
    tree = ast.parse(source, filename=label)
    found: list[StubBody] = []

    def walk(node: ast.AST, scope: tuple[str, ...], owner: ast.ClassDef | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, scope + (child.name,), child)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                form = _stub_form(child.body)
                if form is not None:
                    exempt = None
                    if _is_abstract(child):
                        exempt = "abstractmethod"
                    elif _is_protocol(owner):
                        exempt = "protocol"
                    found.append(
                        StubBody(
                            path=label,
                            qualname=".".join(scope + (child.name,)),
                            line=child.lineno,
                            form=form,
                            exempt_as=exempt,
                        )
                    )
                walk(child, scope + (child.name,), None)
            else:
                walk(child, scope, owner)

    walk(tree, (), None)
    return found


def _stub_bodies_of(path: Path) -> list[StubBody]:
    return stub_bodies(path.read_text(encoding="utf-8"), _relative(path))


# --------------------------------------------------------------------------- #
# Property 35 -- delivered sources contain no placeholders (Requirement 11.8)
# --------------------------------------------------------------------------- #
#
# Finite quantifier again, over the delivered source files, and driven the same
# way: `sampled_from` exhausts the pool at 100 examples and shrinks a failure to
# one named file, with an exhaustive loop alongside so coverage does not rest on
# the pool staying small.


# Feature: ai-room-tile-visualizer, Property 35: Delivered sources contain no
# placeholders
@_PROPERTY_SETTINGS
@given(path=st.sampled_from(TEXT_SOURCES))
def test_property_35_delivered_sources_contain_no_placeholders(path):
    """For any delivered Python or JavaScript source, the file contains no
    placeholder marker standing in for required behaviour, and no function body
    consists solely of `pass` or a raised `NotImplementedError` other than in a
    declared abstract method of the Segmenter interface.

    Requirement 11.8 is a delivery gate: a marker is a note that the author knew
    something was missing, and an empty body is the missing thing itself. Both
    are invisible to every behavioural test, because code that was never written
    is code no test calls.

    Interface declarations are the one exception, and it is narrow: an
    `@abstractmethod` on the Segmenter ABC and a member of a `typing.Protocol`
    are contracts whose body is *supposed* to be empty. The exempt set is pinned
    to an exact inventory in the test below, so the exception cannot widen
    quietly.

    **Validates: Requirements 11.8**
    """
    text = path.read_text(encoding="utf-8")
    hits = marker_hits(text)
    assert not hits, f"{_relative(path)} carries placeholder markers: {hits}"

    if path.suffix == ".py":
        stubs = [stub for stub in _stub_bodies_of(path) if stub.exempt_as is None]
        assert not stubs, f"placeholder function bodies: {[str(stub) for stub in stubs]}"


def test_no_delivered_source_carries_a_placeholder_marker():
    """Property 35's marker half over every delivered source at once, reporting
    all offenders together rather than shrinking to one."""
    offenders = {
        _relative(path): marker_hits(path.read_text(encoding="utf-8"))
        for path in TEXT_SOURCES
        if marker_hits(path.read_text(encoding="utf-8"))
    }
    assert not offenders, f"placeholder markers found: {offenders}"


def test_the_only_stub_bodies_are_declared_interface_members():
    """Property 35's body half, as an exact inventory rather than a filter.

    Listing the exempt declarations by name is the point. A test that merely
    skipped anything decorated `@abstractmethod` would wave through a new
    half-written method the moment someone reached for that decorator; this fails
    on any addition, exempt or not, and makes accepting one a deliberate edit.
    """
    stubs = [stub for path in PYTHON_SOURCES for stub in _stub_bodies_of(path)]

    unexpected = [str(stub) for stub in stubs if stub.exempt_as is None]
    assert not unexpected, f"placeholder function bodies: {unexpected}"

    inventory = sorted((stub.path, stub.qualname, stub.exempt_as) for stub in stubs)
    assert inventory == [
        ("backend/core/segmenter.py", "InferenceSessionLike.get_inputs", "protocol"),
        ("backend/core/segmenter.py", "InferenceSessionLike.get_outputs", "protocol"),
        ("backend/core/segmenter.py", "InferenceSessionLike.run", "protocol"),
        ("backend/core/segmenter.py", "Segmenter.backend_name", "abstractmethod"),
        ("backend/core/segmenter.py", "Segmenter.segment", "abstractmethod"),
    ], f"the set of empty-bodied declarations changed: {inventory}"


def test_the_delivered_source_inventory_is_complete():
    """Guard for Property 35: an empty pool would pass it vacuously.

    Pins that the walk reaches all three Python roots and the frontend module
    directory, and that the specific modules the requirements name by path are
    inside what was scanned.
    """
    scanned = {_relative(path) for path in TEXT_SOURCES}

    assert len(PYTHON_SOURCES) >= 20, f"only {len(PYTHON_SOURCES)} Python sources found"
    for expected in (
        "backend/app.py",
        "backend/core/segmenter.py",
        "backend/utils/model_loader.py",
        "scripts/setup_assets.py",
        "tests/conftest.py",
        "tests/fixtures/synthetic.py",
        "tests/test_static_quality.py",
        "frontend/js/api.js",
        "frontend/js/visualizer.js",
        "frontend/css/visualizer.css",
        "frontend/index.html",
    ):
        assert expected in scanned, f"{expected} was not scanned"

    # Installed and generated trees must not be in scope: pulling `.venv` in
    # would make the scan report third-party markers as project defects.
    assert not any(part.startswith(".") for path in TEXT_SOURCES for part in
                   path.relative_to(_PROJECT_ROOT).parts)


# --------------------------------------------------------------------------- #
# Placeholder detector guards
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("marker", MARKERS, ids=lambda marker: marker.replace(" ", "_"))
def test_every_marker_is_detected_in_a_comment(marker):
    """Each marker in the vocabulary must actually be matched. Built from
    `MARKERS` so the assertion cannot drift from the pattern it tests."""
    assert marker_hits(f"x = 1  # {marker}: finish this\n")
    assert marker_hits(f"// {marker.upper()} handle the error case\n")
    assert marker_hits(f"y = 2  # {marker.lower()}\n")


def test_marker_matching_is_word_bounded():
    """Prose must not be flagged. A check that fired on `wipes` or `hackathon`
    would be turned off by the next person to hit it, so the boundaries are
    pinned as tightly as the detection."""
    for benign in (
        "cv2.remap wipes the destination buffer\n",
        "# the shading map is computed, not approximated\n",
        "# every plane is absent from the mapping rather than present with a placeholder\n",
        "# stubbed sessions speak the SAM point-prompt contract\n",
        "# 600x600 and 600x1200 formats\n",
        "# tbdisplay = False\n",
    ):
        assert not marker_hits(benign), f"{benign!r} was wrongly flagged"


def test_marker_matching_spans_a_wrapped_comment():
    """A marker split across two comment lines is still a marker, so the
    multi-word ones tolerate a line break where their space is."""
    wrapped = "# the neural path is not\n# implemented on this host\n"
    assert marker_hits(wrapped)


@pytest.mark.parametrize(
    ("source", "form"),
    [
        ("def f():\n    pass\n", "pass"),
        ("def f():\n    ...\n", "ellipsis"),
        ('def f():\n    """Doc."""\n    pass\n', "pass"),
        ("def f():\n    raise NotImplementedError\n", "raise NotImplementedError"),
        ("def f():\n    raise NotImplementedError('later')\n", "raise NotImplementedError"),
        ("async def f():\n    pass\n", "pass"),
        ("class C:\n    def f(self):\n        pass\n", "pass"),
        ("def outer():\n    def inner():\n        pass\n    return inner\n", "pass"),
    ],
    ids=[
        "pass",
        "ellipsis",
        "docstring_then_pass",
        "raise_bare",
        "raise_called",
        "async_def",
        "method",
        "nested_function",
    ],
)
def test_the_stub_detector_reports_known_placeholder_bodies(source, form):
    """Every placeholder shape the property names, plus the shapes that would
    hide one: a docstring in front of it, `async def`, and nesting."""
    stubs = [stub for stub in stub_bodies(source, "synthetic.py") if stub.exempt_as is None]
    assert [stub.form for stub in stubs] == [form]


@pytest.mark.parametrize(
    "source",
    [
        "def f():\n    return 1\n",
        'def f():\n    """Doc."""\n    return 1\n',
        "def f():\n    try:\n        g()\n    except OSError:\n        pass\n",
        "def f():\n    for x in y:\n        if x:\n            pass\n    return y\n",
        "def f():\n    raise ValueError('bad input')\n",
        "class C:\n    x: int = 1\n",
    ],
    ids=[
        "real_body",
        "docstring_and_body",
        "pass_inside_an_except_handler",
        "pass_inside_a_loop",
        "raise_a_real_error",
        "class_with_no_methods",
    ],
)
def test_the_stub_detector_ignores_real_implementations(source):
    """`pass` as a statement inside a handler or a loop is control flow, not a
    placeholder. Three files in `backend/` swallow an exception that way, so a
    detector that could not tell the difference would be unusable."""
    assert stub_bodies(source, "synthetic.py") == []


@pytest.mark.parametrize(
    ("source", "exempt_as"),
    [
        (
            "from abc import ABC, abstractmethod\n"
            "class S(ABC):\n"
            "    @abstractmethod\n"
            "    def segment(self):\n"
            "        ...\n",
            "abstractmethod",
        ),
        (
            "from abc import ABC, abstractmethod\n"
            "class S(ABC):\n"
            "    @property\n"
            "    @abstractmethod\n"
            "    def backend_name(self):\n"
            "        ...\n",
            "abstractmethod",
        ),
        (
            "from typing import Protocol\n"
            "class P(Protocol):\n"
            "    def run(self):\n"
            "        ...\n",
            "protocol",
        ),
    ],
    ids=["abstract_method", "abstract_property", "protocol_member"],
)
def test_the_stub_detector_classifies_interface_declarations_as_exempt(source, exempt_as):
    """The exempt forms are recognised as exempt, and still reported, which is
    what lets the inventory test above see them."""
    stubs = stub_bodies(source, "synthetic.py")
    assert [stub.exempt_as for stub in stubs] == [exempt_as]


def test_an_undecorated_stub_in_an_abstract_class_is_not_exempt():
    """Exemption follows the decorator, not the class. A concrete method sitting
    beside abstract ones is ordinary code and must be written."""
    source = (
        "from abc import ABC, abstractmethod\n"
        "class S(ABC):\n"
        "    @abstractmethod\n"
        "    def segment(self):\n"
        "        ...\n"
        "    def helper(self):\n"
        "        pass\n"
    )
    stubs = stub_bodies(source, "synthetic.py")
    assert [(stub.qualname, stub.exempt_as) for stub in stubs] == [
        ("S.segment", "abstractmethod"),
        ("S.helper", None),
    ]


# --------------------------------------------------------------------------- #
# Dependency manifest (Requirement 11.6)
# --------------------------------------------------------------------------- #

#: The line that opens the development and test section. Requirement 11.6 asks
#: for a *labelled* section, so the label is matched rather than the position.
#: Anchored past only decoration -- `# --- development / test dependencies ---`
#: is a label, while the file header's prose sentence mentioning development and
#: test dependencies is not, and must not be mistaken for the divider.
_DEV_SECTION_RE = re.compile(r"^#[\s\-=*_]*(development|dev)\b", re.IGNORECASE)

#: `name`, an optional extras list, then an exact-version pin. Anything looser --
#: `>=`, `~=`, a bare name -- fails to match and is reported.
_PIN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?==[A-Za-z0-9.*+!-]+$")

_EXPECTED_RUNTIME = (
    "fastapi",
    "uvicorn",
    "python-multipart",
    "pydantic",
    "pydantic-settings",
    "numpy",
    "opencv-python",
    "onnxruntime",
    "httpx",
)
_EXPECTED_DEV = ("pytest", "pytest-cov", "hypothesis", "psutil")


def _requirement_sections() -> tuple[list[str], list[str], bool]:
    """Split the manifest into runtime and development requirement lines."""
    runtime: list[str] = []
    development: list[str] = []
    seen_label = False
    for raw in _REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if _DEV_SECTION_RE.match(line) and "test" in line.lower():
                seen_label = True
            continue
        (development if seen_label else runtime).append(line)
    return runtime, development, seen_label


def _distribution_name(line: str) -> str:
    return re.split(r"[\[=<>~!;]", line, maxsplit=1)[0].strip().lower()


def test_every_runtime_dependency_is_pinned_to_an_exact_version():
    """Requirement 11.6. A range means two clones can install different code and
    only one of them reproduces a bug, which is the failure mode a pinned
    manifest exists to remove."""
    runtime, development, _ = _requirement_sections()

    assert runtime, f"no runtime requirements parsed from {_relative(_REQUIREMENTS_PATH)}"
    for line in runtime + development:
        assert _PIN_RE.match(line), f"{line!r} is not an exact `==` pin"
        assert not re.search(r"[<>~]=|(?<![=!<>])>|(?<![=!<>])<", line), (
            f"{line!r} carries a range specifier"
        )


def test_the_manifest_declares_a_labelled_development_and_test_section():
    """Requirement 11.6's second half: the dev and test dependencies are present,
    and separated by a label rather than by convention, so `pip install -r` sets
    up both the service and the suite from one file."""
    runtime, development, seen_label = _requirement_sections()
    assert seen_label, "no labelled development / test section found"
    assert development, "the development / test section declares nothing"

    runtime_names = {_distribution_name(line) for line in runtime}
    dev_names = {_distribution_name(line) for line in development}

    assert set(_EXPECTED_RUNTIME) <= runtime_names, (
        f"missing runtime dependencies: {set(_EXPECTED_RUNTIME) - runtime_names}"
    )
    assert set(_EXPECTED_DEV) <= dev_names, (
        f"missing development dependencies: {set(_EXPECTED_DEV) - dev_names}"
    )
    # The split has to be real: a test dependency listed above the label would be
    # installed as a runtime requirement.
    assert not runtime_names & dev_names
    assert not dev_names & set(_EXPECTED_RUNTIME)


def test_the_pin_check_rejects_loose_specifiers():
    """Guard: the pattern must actually reject what it is there to catch."""
    for loose in ("fastapi", "fastapi>=0.141", "numpy~=2.3", "httpx<1.0", "pydantic!=2.0"):
        assert not _PIN_RE.match(loose), f"{loose!r} was accepted as an exact pin"
    for pinned in ("fastapi==0.141.1", "uvicorn[standard]==0.52.4", "pytest-cov==7.1.0"):
        assert _PIN_RE.match(pinned), f"{pinned!r} was rejected as an exact pin"


# --------------------------------------------------------------------------- #
# README (Requirements 2.8, 9.7, 11.7)
# --------------------------------------------------------------------------- #

README_TEXT: str = _README_PATH.read_text(encoding="utf-8")

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.*?)\s*$", re.M)


def _readme_sections() -> dict[str, str]:
    """Heading text (lowercased, markup stripped) mapped to its body.

    A section runs to the next heading at the same or a higher level, so it
    carries its subsections with it: `## Environment variables` is a page of
    `###` tables, and a body that stopped at the first subheading would be empty
    and every assertion over it vacuous.
    """
    headings = list(_HEADING_RE.finditer(README_TEXT))
    sections: dict[str, str] = {}
    for index, match in enumerate(headings):
        level = len(match.group(1))
        end = len(README_TEXT)
        for following in headings[index + 1 :]:
            if len(following.group(1)) <= level:
                end = following.start()
                break
        title = match.group(2).replace("`", "").replace("*", "").strip().lower()
        sections.setdefault(title, README_TEXT[match.end() : end])
    return sections


README_SECTIONS: dict[str, str] = _readme_sections()


def _section_matching(*needles: str) -> tuple[str, str]:
    """The first section whose heading contains every needle, with its title."""
    for title, body in README_SECTIONS.items():
        if all(needle.lower() in title for needle in needles):
            return title, body
    raise AssertionError(f"no README heading contains all of {needles}: {list(README_SECTIONS)}")


@pytest.mark.parametrize(
    "needles",
    [
        ("installation",),
        ("setup",),
        ("running", "service"),
        ("http api",),
        ("embedding", "frontend"),
        ("environment variables",),
        ("security",),
        ("deployment",),
    ],
    ids=[
        "installation",
        "asset_setup",
        "running_the_service",
        "http_api",
        "embedding",
        "environment_variables",
        "security",
        "deployment",
    ],
)
def test_the_readme_documents_every_required_topic(needles):
    """Requirement 11.7 enumerates what the README owes a developer: how to
    install, how to set up and run, the endpoints, how to embed, and the
    configuration variables. Requirements 2.8 and 9.7 add the security posture
    and the single-worker constraint."""
    title, body = _section_matching(*needles)
    assert body.strip(), f"README section {title!r} is empty"


@pytest.mark.parametrize(
    ("endpoint", "method"),
    [
        ("/api/segment", "POST"),
        ("/api/render", "POST"),
        ("/api/tiles", "GET"),
        ("/api/health", "GET"),
    ],
)
def test_the_readme_documents_every_endpoint_with_its_shapes(endpoint, method):
    """Requirement 11.7: every HTTP endpoint, with request and response shapes.

    A heading per endpoint and a JSON block in its section, so the documented
    contract is concrete enough to code against. The two POST endpoints also
    have to describe what to send, since neither is callable without a body.
    """
    title, body = _section_matching(endpoint)
    assert method.lower() in title, f"section {title!r} does not name the {method} method"
    assert "```json" in body, f"section {title!r} shows no JSON response shape"
    if method == "POST":
        assert "request" in body.lower(), f"section {title!r} does not describe the request"


def test_the_readme_documents_every_environment_variable():
    """Requirement 11.7: the configuration variables, all of them.

    Derived from `Settings` rather than from a hand-written list, so adding a
    field to the settings object and forgetting to document it fails here
    instead of leaving an operator to read the source.
    """
    _, body = _section_matching("environment variables")
    documented = set(re.findall(r"RV_[A-Z0-9_]+", README_TEXT))
    expected = {f"RV_{field.upper()}" for field in Settings.model_fields}

    assert expected <= documented, f"undocumented settings: {sorted(expected - documented)}"
    assert not documented - expected, f"README documents unknown variables: {sorted(documented - expected)}"
    # The variables belong in the section that claims to list them, not scattered
    # through prose elsewhere.
    in_section = set(re.findall(r"RV_[A-Z0-9_]+", body))
    assert expected <= in_section, (
        f"settings documented outside the environment variables section: "
        f"{sorted(expected - in_section)}"
    )


def test_the_readme_states_the_service_is_unauthenticated_and_needs_a_proxy():
    """Requirement 2.8, verbatim in substance: no authentication, no rate
    limiting, and a public deployment needs an authenticating reverse proxy in
    front. This is the one README section whose absence is a security defect
    rather than a documentation gap."""
    _, security = _section_matching("security")
    lowered = security.lower()

    assert "no authentication" in lowered
    assert "no rate limiting" in lowered
    assert "reverse proxy" in lowered
    assert "authenticating reverse proxy" in lowered
    for expected in ("rate limiting", "request-size"):
        assert expected in lowered, f"the proxy guidance does not mention {expected}"


def test_the_readme_documents_the_single_worker_constraint():
    """Requirement 9.7: the cache is process-local, scene state is lost on
    restart, and the service runs with one uvicorn worker unless a shared cache
    is introduced. All three, because each one alone leaves an operator able to
    break tile swapping in a way that looks like a bug in the product."""
    _, deployment = _section_matching("deployment")
    lowered = deployment.lower()

    assert "--workers 1" in deployment
    assert "process-local" in lowered
    assert "lost on process restart" in lowered
    assert "shared cache" in lowered
    assert "scene_expired" in lowered

    # The run instructions must not contradict the deployment guidance.
    _, running = _section_matching("running", "service")
    assert "--workers 1" in running


def test_the_readme_shows_the_component_embedding_call():
    """Requirement 11.7 and 10.1: one script, one constructor call, no build
    step. The example has to be the real API surface, not prose about it."""
    _, embedding = _section_matching("embedding", "frontend")

    assert "new RoomVisualizer(" in embedding
    assert "apiBaseUrl" in embedding
    assert "visualizer.js" in embedding
    assert "visualizer.css" in embedding
    assert "window.RoomVisualizer" in embedding


def test_the_readme_section_index_is_usable():
    """Guard for the README tests: `_section_matching` searching an empty or
    single-entry index would make every assertion above vacuous."""
    assert len(README_SECTIONS) > 15, f"only {len(README_SECTIONS)} headings parsed"
    with pytest.raises(AssertionError):
        _section_matching("a heading the readme certainly does not have")


# --------------------------------------------------------------------------- #
# Mandated layout (Requirement 11.1)
# --------------------------------------------------------------------------- #

#: Requirement 11.1, transcribed. These are the paths the requirement names
#: literally; the supporting modules around them are the design's own additions
#: and are deliberately not pinned here, so the layout can grow but cannot lose
#: a mandated file or move one to a different name.
_MANDATED_PATHS = (
    "backend/app.py",
    "backend/requirements.txt",
    "backend/core/segmenter.py",
    "backend/core/geometry.py",
    "backend/core/lighting.py",
    "backend/core/compositor.py",
    "backend/utils/texture_helper.py",
    "backend/utils/model_loader.py",
    "frontend/index.html",
    "frontend/css/visualizer.css",
    "frontend/js/visualizer.js",
    "frontend/js/api.js",
    "README.md",
)


@pytest.mark.parametrize("relative", _MANDATED_PATHS)
def test_every_mandated_project_path_exists_and_is_not_empty(relative):
    """Requirement 11.1 fixes these thirteen paths by name. An embedder follows
    them from the README, so a rename is a broken contract even when the code
    behind it still works."""
    path = _PROJECT_ROOT / relative
    assert path.is_file(), f"Requirement 11.1 mandates {relative}, which is missing"
    assert path.stat().st_size > 0, f"{relative} is empty"


def test_the_mandated_path_list_matches_the_requirement():
    """Guard: thirteen paths are named in Requirement 11.1. A transcription that
    quietly lost one would make the check above pass while the layout drifted."""
    assert len(_MANDATED_PATHS) == 13
    assert len(set(_MANDATED_PATHS)) == 13
