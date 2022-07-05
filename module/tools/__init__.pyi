from collections.abc import Generator, Iterable
from typing import TypeVar
T = TypeVar('T')


def flatten_generator(lst: Iterable[T]) -> Generator[T, None, None]: ...


def flatten(lst: Iterable[T]) -> list[T]: ...


def linear_solve(a: float, b: float) -> float: ...


def quadratic_solve(a: float, b: float, c: float, avg: float, fit: bool) -> float: ...


def cubic_solve(a: float, b: float, c: float, d: float, avg: float, fit: bool) -> float: ...


def quartic_solve(a: float, b: float, c: float, d: float, e: float, avg: float, fit: bool) -> float: ...


def ValErr(value: int, expected: int, singular: str, plural: str) -> None | ValueError: ...
