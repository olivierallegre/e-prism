import numpy as np


def flatten(seq):
    l = []
    for elt in seq:
        t = type(elt)
        if t is tuple or t is list:
            for elt2 in flatten(elt):
                l.append(elt2)
        else:
            l.append(elt)
    return l


def get_set_of_knowledge_components_from_dataset(dataset, defaults="kartable"):
    import pandas as pd
    from .exercises import Exercise
    assert any((isinstance(dataset, str), isinstance(dataset, pd.DataFrame))), "dataset must be str or dataframe"

    if not isinstance(defaults, dict):
        if defaults == "kartable":
            defaults = {'order_id': 'uea_id', 'skill_name': 'kd_id', 'kc_id': 'kd_id', 'exercise_id': 'kae_id',
                        'correct': 'uea_success',
                        'user_id': 'user_id', 'multilearn': 'kd_id', 'multiprior': 'uea_success',
                        'multipair': 'kd_id', 'multigs': 'kae_id', 'folds': 'user_id'}
        elif defaults == 'standard':
            defaults = {'order_id': 'idx', 'skill_name': 'kc_id', 'kc_id': 'kc_id', 'exercise_id': 'exercise_id',
                        'correct': 'correct',
                        'user_id': 'learner_id', 'multilearn': 'kc_id', 'multiprior': 'correct',
                        'multipair': 'kc_id', 'multigs': 'exercise_id', 'folds': 'learner_id'}
        elif defaults == 'as':
            defaults = {'order_id': 'order_id', 'skill_name': 'skill_name', 'kc_id': 'skill_id',
                        'exercise_id': 'item_id',
                        'correct': 'correct',
                        'user_id': 'user_id', 'multilearn': 'skill_id', 'multiprior': 'correct',
                        'multipair': 'skill_id', 'multigs': 'problem_id', 'folds': 'user_id'}

    df = dataset if isinstance(dataset, pd.DataFrame) else pd.read_csv(dataset)
    knowledge_components, exercises = [], []
    for i, row in df.iterrows():
        kc_id = row[defaults['kc_id']]
        if kc_id not in [kc.id for kc in knowledge_components]:
            kc_name = row[defaults['skill_name']] if defaults['skill_name'] in row.keys() else kc_id
            kc = KnowledgeComponent(kc_id, kc_name)
            knowledge_components.append(kc)
        else:
            kc = [kc for kc in knowledge_components if kc.id == kc_id][0]
        exercise_id = row[defaults['exercise_id']]
        if exercise_id not in [exercise.id for exercise in exercises]:
            exercise = Exercise(row[defaults['exercise_id']], kc)
            exercises.append(exercise)
    return SetOfKnowledgeComponents(knowledge_components)


class KnowledgeComponent(object):

    def __init__(self, kc_id: int, kc_name: str, exercises=None):
        """
        Constructor of the KnowledgeComponent class
        :param kc_id: the id of the KnowledgeComponent
        :param kc_name: the name of the KnowledgeComponent
        :param exercises: exercises associated to the KC
        """
        self.id = int(kc_id)
        self.name = str(kc_name)
        self.exercises = [] if exercises is None else exercises
        self.behavior = None

    def add_associated_exercise(self, exercise):
        """
        Add an exercise to associated exercises pool.
        :param exercise: exercise to add
        """
        from .exercises import Exercise
        assert isinstance(exercise, Exercise), "Exercise object expected."

        self.exercises.append(exercise)
        if not exercise.get_kc() is self:
            exercise.set_kc(self)
        exercise.set_kc(self)

    def get_exercises(self):
        """
        Get exercises associated to the KC.
        :return: list of exercises associated to the KC
        """
        return self.exercises

    def get_random_exercise(self):
        import random
        return self.exercises[random.randint(0, len(self.exercises)-1)]


class SetOfKnowledgeComponents(object):

    def __init__(self, knowledge_components=None):
        """
        The class Domain corresponds to the model of the expert knowledge on the learning domain. It is composed
        of two elements: knowledge components (elements that compose the expected knowledge -- overlay model) and the
        prerequisite links
        """
        if knowledge_components is None:
            knowledge_components = []
        self.knowledge_components = knowledge_components

    def __str__(self):
        return [kc.name for kc in self.knowledge_components]

    def add_kc(self, kc: KnowledgeComponent):
        """
        Add a given knowledge component into the Domain's knowledge components.
        :param kc: the knowledge component to be added
        """
        assert isinstance(kc, KnowledgeComponent), "Entered KC is not a KnowledgeComponent object."
        if kc not in self.knowledge_components:
            self.knowledge_components.append(kc)

    def remove_kc(self, kc):
        """
        Remove a given knowledge component from the Domain's knowledge components.
        :param kc: the knowledge component to be removed
        """
        if isinstance(kc, (list, np.ndarray)):
            for k_el in kc:
                self.knowledge_components.remove(k_el)
        else:
            self.knowledge_components.remove(kc)

    def get_knowledge_components(self):
        """
        Return all Domain's knowledge components.
        :return: the list of Domain's knowledge components
        """
        return self.knowledge_components

    def get_set_of_exercises(self):
        from .exercises import SetOfExercises
        return SetOfExercises(flatten([kc.get_exercises() for kc in self.knowledge_components]))

    def get_kc_from_name(self, kc_name):
        knowledge_components = [kc for kc in self.knowledge_components if kc.name == kc_name]
        if len(knowledge_components) > 0:
            return knowledge_components[0]
        else:
            return Exception()

    def get_kc_from_id(self, kc_id):
        knowledge_components = [kc for kc in self.knowledge_components if int(kc.id) == kc_id]
        if len(knowledge_components) > 0:
            return knowledge_components[0]
        else:
            return None

    def get_random_kc(self):
        import random
        return self.knowledge_components[random.randint(0, len(self.knowledge_components)-1)]

