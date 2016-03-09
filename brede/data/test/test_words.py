"""Test of words module."""


import pytest

from .. import words


@pytest.fixture
def task_to_words():
    ttw = words.TaskToWords()
    return ttw


def test_task_to_words_tasks(task_to_words):
    tasks = task_to_words.tasks
    assert len(tasks) >= 5
    assert 'Left hand sequential finger tapping' in tasks
