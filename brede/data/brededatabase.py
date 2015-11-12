"""brede.data.brededatabase - interface to data from Brede Database.

Usage:
  brede.data.brededatabase [options]

Options:
  -h --help  Help

"""

from __future__ import absolute_import, division, print_function

from os.path import dirname, join

from pandas import read_csv


class TaskToCognitiveComponent(object):
    """Represents cognitive components of tasks.

    Data is read from a comma-separated file.

    """

    def __init__(self):
        """Setup filename."""
        self.filename = join(dirname(__file__),
                             'brede_database_data',
                             'task_to_cognitive_component.csv')
        self._data = self.load_data()

    def load_data(self):
        """Load and return data from file."""
        return read_csv(self.filename)

    def scores_for_task(self, task):
        """Return scores for cognitive components wrt. task.

        Returns
        -------
        scores : dict
            Dictionary with cognitive components as keys and scores as values

        Examples
        --------
        >>> ttcc = TaskToCognitiveComponent()
        >>> scores = ttcc.scores_for_task('Face viewing')
        >>> scores['Face recognition']
        5

        """
        df = self._data.ix[self._data['Task'] == task, :]
        return df.set_index('Cognitive component')['Score'].to_dict()

    @property
    def tasks(self):
        """Return unique tasks.

        Returns
        -------
        tasks : list of strings

        Examples
        --------
        >>> ttcc = TaskToCognitiveComponent()
        >>> 'Left hand sequential finger tapping' in ttcc.tasks
        True

        """
        return list(self._data['Task'].unique())


def main(args):
    """Handle command-line interface."""
    ttcc = TaskToCognitiveComponent()
    print(ttcc.tasks)


if __name__ == '__main__':
    import docopt

    main(docopt.docopt(__doc__))
