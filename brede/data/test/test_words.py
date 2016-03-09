"""Test of words module."""


import pytest

from .. import words


@pytest.fixture
def task_to_words():
    """Setup TastToWork object."""
    ttw = words.TaskToWords()
    return ttw


def test_task_to_words_tasks(task_to_words):
    """Test task property in TaskToWords."""
    tasks = task_to_words.tasks
    assert len(tasks) >= 5
    assert 'Left hand sequential finger tapping' in tasks


def test_task_to_words_score(task_to_words):
    """Test scores.

    Scores should be integers between 0 and 5.
    """
    tasks = task_to_words.tasks
    for task in tasks:
        scores = task_to_words.scores_for_task(task)
        for score in scores.values():
            assert score == int(score)
            assert score >= 0 and score <= 5
