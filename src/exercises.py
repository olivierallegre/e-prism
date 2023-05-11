import json
from ast import literal_eval


class Exercise:

    def __init__(self, ex_id, knowledge_component, ex_type=None, ex_content=None, params=None):
        """
        Initialization of Exercise object
        :param ex_id: int, exercise id
        :param ex_content: str, exercise content
        :param params: dict, exercise parameters -- keys must belong to [learn, guess, slip, delta, gamma]
        """
        self.id = ex_id
        self.set_kc(knowledge_component)

        if not ex_content:
            self.content = "Empty"
        elif ex_content[1] == "'":
            self.content = literal_eval(ex_content)
        else:
            self.content = json.loads(ex_content)
        assert self.content, print(f"Exercise #{self.id} content is empty.")
        self.type = ex_type
        self.params = {}

        if params is not None:
            if "guess" in params.keys():
                self.params["guess"] = params["guess"]
            if "slip" in params.keys():
                self.params["slip"] = params["slip"]

    def set_slip(self, slip):
        """
        Set the value of the slip parameter with a given value.
        :param slip: the wanted value for slip parameter
        :return: None, only sets the value to param value
        """
        self.params['slip'] = slip

    def get_slip(self):
        """
        Get the value of the slip parameter of self Exercise.
        :return: slip parameter of self
        """
        return self.params['slip']

    def set_guess(self, guess):
        """
        Set the value of the guess parameter with a given value.
        :param guess: the wanted value for guess parameter
        :return: None, only sets the value to param value
        """
        self.params['guess'] = guess

    def get_guess(self):
        """
        Get the value of the guess parameter of self Exercise.
        :return: guess parameter of self
        """
        return self.params['guess']

    def set_kc(self, kc):
        from .knowledge_components import KnowledgeComponent
        assert isinstance(kc, KnowledgeComponent), "KnowledgeComponent object expected."
        self.knowledge_component = kc
        if self not in self.knowledge_component.get_exercises():
            self.knowledge_component.add_associated_exercise(self)

    def get_kc(self):
        """
        Get the knowledge component object associated to self Exercise.
        :return: knowledge component object associated to self
        """
        return self.knowledge_component


class SetOfExercises:

    def __init__(self, exercises):
        assert isinstance(exercises, list), "List expected."
        for elt in exercises:
            assert isinstance(elt, Exercise), "All elements of SetOfExercises must be exercises object"

        self.exercises = exercises

    def get_exercise_from_id(self, exercise_id):
        results = [ex for ex in self.exercises if ex.id == exercise_id]
        if len(results) > 0:
            return results[0]
        else:
            return Exception('No exercise with given id')

    def get_exercise_ids(self):
        return [exercise.id for exercise in self.exercises]
