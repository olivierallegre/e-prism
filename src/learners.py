import numpy as np
import pandas as pd
import itertools
from .inference_models import NoisyANDInferenceModel, NoisyORInferenceModel, LeakyORInferenceModel, \
    LeakyANDInferenceModel, SimpleORInferenceModel
from .knowledge_components import KnowledgeComponent, SetOfKnowledgeComponents
from .exercises import Exercise
from .prerequisite_links import PrerequisiteLink, SetOfPrerequisiteLinks
from .utils import gibbs_sample_to_em_data, truthtable, flatten, unrolled_sample_to_2tbn_samples
import re
import tqdm
import pyAgrum as gum


class LearnerTrace:

    def __init__(self, learner, exercise, success):
        """
        Init of the LearnerTrace object
        :param learner: Learner object, learner that has done the trace
        :param exercise: Exercise object, exercise on which is the trace
        :param success: bool, success of the learner on the exercise
        """
        assert isinstance(learner, Learner), "Learner must be a mskt Learner object"
        self.learner = learner
        assert isinstance(exercise, Exercise), "Exercise must be a mskt Exercise object"
        self.exercise = exercise
        assert isinstance(success, bool), "Success must be a boolean."
        self.success = success
        self.knowledge_component = self.exercise.get_kc()

    def get_kc(self):
        """
        Get the KC that has been practiced by the learner trace.
        :return: KnowledgeComponent
        """
        return self.knowledge_component

    def get_success(self):
        """
        Get the success of the learner trace.
        :return: bool, success
        """
        return self.success

    def get_exercise(self):
        """
        Get the exercise on which rely the learner trace.
        :return: Exercise
        """
        return self.exercise

    def get_learner(self):
        """
        Get the learner that has produced the learner trace.
        :return: Learner
        """
        return self.learner


class Learner:

    def __init__(self, learner_id, learner_pool=None):
        """
        Initialization of Learner object.
        :param learner_id: id of the learner
        :param learner_pool: LearnerPool, the LearnerPool of which the learner belongs
        """
        self.id = learner_id
        self.learner_pool = learner_pool
        if self.learner_pool:
            self.learner_pool.add_learner(self)
            knowledge_components = self.learner_pool.get_knowledge_components()
            self.mastering_probabilities = {kc: self.learner_pool.get_prior(kc) for kc in knowledge_components}

    def set_learner_pool(self, learner_pool):
        """
        Set a new learner pool associated to the learner
        :param learner_pool: LearnerPool, new learner pool
        """
        self.__init__(self.id, learner_pool)

    def set_mastering_probability(self, kc, mastering_probability):
        """
        Set the mastering probability of a given kc to a new value
        :param kc: KnowledgeComponent, the KC on which is the new mastering probability
        :param mastering_probability: the new mastering probability value
        """
        assert kc in self.mastering_probabilities.keys()
        assert 0 <= mastering_probability <= 1, "Mastering probability must be a float in [0, 1]."
        self.mastering_probabilities[kc] = mastering_probability

    def get_mastering_probability(self, kc):
        """
        Get the mastering probability of the learner on a given KC.
        :param kc: KnowledgeComponent
        :return: the value of the mastering probability
        """
        return self.mastering_probabilities[kc]

    def get_predicted_answers_from_learner_traces(self, learner_traces, mode='other'):
        """
        Return the sequence of predicted answers on given learner traces. For the nth learner trace, the model will take
        into account the n-1 traces that has been done before and will predict the success of the nth learner trace.
        :param learner_traces: learner traces on which the prediction is expected
        :param inference_model_type: the type of the inference model (for now, two possible choices: Noisy-AND and
        Noisy-OR)
        :return: list, the list of the predicted answers for every learner traces
        """
        inference_model = self.learner_pool.inference_model

        if isinstance(inference_model, LeakyANDInferenceModel):
            correct_predictions = [self.learner_pool.inference_model.get_posterior_given_evidence(
                node=f"(O[{trace.get_kc().id}]){i}",
                evs={f"(O[{ev_trace.get_kc().id}]){ev_i}": ev_trace.get_success() for ev_i, ev_trace in
                     enumerate(learner_traces[:i])}
            ) for i, trace in enumerate(learner_traces)]

        if mode == 'all':
            correct_predictions = [self.learner_pool.inference_model.get_posterior_given_evidence(
                node=f"({trace.get_kc().id}){i}",
                evs={f"({ev_trace.get_kc().id}){ev_i}": ev_trace.get_success() for ev_i, ev_trace in
                     enumerate(learner_traces) if ev_i != i}
            ) for i, trace in enumerate(learner_traces)]
        else:
            correct_predictions = [self.learner_pool.inference_model.get_posterior_given_evidence(
                node=f"({trace.get_kc().id}){i}",
                evs={f"({ev_trace.get_kc().id}){ev_i}": ev_trace.get_success() for ev_i, ev_trace in
                     enumerate(learner_traces[:i])}
            ) for i, trace in enumerate(learner_traces)]

        return correct_predictions

    def simulate_mastery_evolution(self, n_transactions_max, evs={}, return_array=False):
        import matplotlib.pyplot as plt
        import pyAgrum.lib.dynamicBN as gdyn

        knowledge_components = self.learner_pool.get_knowledge_components()

        if return_array:
            unrolled_bn = self.learner_pool.inference_model.get_unrolled_dbn(n_transactions_max)
            ie = gum.LazyPropagation(unrolled_bn)
            ie.setEvidence(evs)
            ie.makeInference()
            res = {kc.id: [ie.posterior(unrolled_bn.idFromName(f"({kc.id}){i}")).toarray()[1] for i in
                           range(n_transactions_max)] for kc in knowledge_components}
            del ie
            del unrolled_bn
            return res

        else:
            twodbn = self.learner_pool.inference_model.get_tbn()
            plt.rcParams['figure.figsize'] = (10, 2)
            gdyn.plotFollow(
                ["(" + str(kc.id) + ")" for kc in knowledge_components],
                twodbn,
                T=n_transactions_max,
                evs=evs
            )


class LearnerPool:

    def __init__(self, set_of_knowledge_components, set_of_prerequisite_links, inference_model_type='NoisyAND',
                 params=None, desc='unspecified'):
        """
        The LearnerPool class is supposed to represent groups of Learners that shares similar characteristics. They
        belong on a given domain on which they behave in the same way. We declare a default_learner that will emphasize
        the behavior of a random learner of this pool.
        :param domain : the domain on which the learner of the LearnerPool study.
        """
        self.desc = desc
        self.learners = []

        if isinstance(set_of_knowledge_components, list):
            set_of_knowledge_components = SetOfKnowledgeComponents(set_of_knowledge_components)

        assert isinstance(set_of_knowledge_components, SetOfKnowledgeComponents), f"Unknown given type for knowledge " \
                                                                                  f"components."
        self.set_of_knowledge_components = set_of_knowledge_components

        self.set_of_exercises = self.set_of_knowledge_components.get_set_of_exercises()

        if isinstance(set_of_prerequisite_links, list):
            set_of_prerequisite_links = SetOfKnowledgeComponents(set_of_prerequisite_links)

        assert isinstance(set_of_prerequisite_links, SetOfPrerequisiteLinks), f"Unknown given type for prerequisite " \
                                                                              f"links."
        self.set_of_prerequisite_links = set_of_prerequisite_links

        self._setup_domain_bn()

        self.params = {}
        self.inference_model_type = inference_model_type
        if params is None:
            knowledge_components = set_of_knowledge_components.get_knowledge_components()
            self.params['prior'] = {kc: .2 for kc in knowledge_components}
            self.params['learn'] = {kc: .1 for kc in knowledge_components}
            self.params['forget'] = {kc: .05 for kc in knowledge_components}

            """
            # params in function of items and not kcs
            self.params['slip'] = {x: 0.1 for kc in knowledge_components for x in kc.get_exercises()}
            self.params['guess'] = {x: 0.1 for kc in knowledge_components for x in kc.get_exercises()}
            """

            self.params['slip'] = {kc: .1 for kc in knowledge_components}
            self.params['guess'] = {kc: .1 for kc in knowledge_components}

            self.params['leak'] = {kc: .05 for kc in knowledge_components}
            self.params['c'] = {}
            self.params['c0'] = {}
            c_params = {'strong': .9, 'weak': .7}
            for target_key in self.set_of_prerequisite_links.get_anticausal_map().keys():
                self.params['c'][target_key] = {}
                self.params['c0'][target_key] = {}
                for source_key in self.set_of_prerequisite_links.get_anticausal_map()[target_key].keys():
                    self.params['c'][target_key][source_key] = c_params[
                        self.set_of_prerequisite_links.get_prerequisite_link_strength(source_key, target_key)]
                    self.params['c0'][target_key][source_key] = c_params[
                        self.set_of_prerequisite_links.get_prerequisite_link_strength(source_key, target_key)]

            self.params['s'] = {}
            self.params['s0'] = {}
            s_params = {'strong': .1, 'weak': .3}
            for target_key in self.set_of_prerequisite_links.get_anticausal_map().keys():
                self.params['s'][target_key] = {}
                self.params['s0'][target_key] = {}
                for source_key in self.set_of_prerequisite_links.get_anticausal_map()[target_key].keys():
                    self.params['s'][target_key][source_key] = s_params[
                        self.set_of_prerequisite_links.get_prerequisite_link_strength(source_key, target_key)]
                    self.params['s0'][target_key][source_key] = s_params[
                        self.set_of_prerequisite_links.get_prerequisite_link_strength(source_key, target_key)]
            self._setup_inference_model(self.inference_model_type)
        else:
            self._setup_params(params)

    def _setup_domain_bn(self):
        self.domain_bn = gum.BayesNet()
        knowledge_components = self.get_knowledge_components()
        for kc in knowledge_components:
            # for all knowledge component X, there is a node (X)0...
            self.domain_bn.add(gum.LabelizedVariable(f"{kc.id}", f"{kc.id}", 2))

        for kc in knowledge_components:
            parents = self.get_learner_pool_kc_parents(kc)
            if parents:
                for parent in parents:
                    self.domain_bn.addArc(f"{parent.id}", f"{kc.id}")

    def _setup_inference_model(self, inference_model_type):
        try:
            del self.inference_model
        except:
            pass
        if inference_model_type == 'NoisyOR':
            self._setup_noisyor_inference_model()
        elif inference_model_type == 'NoisyAND':
            self._setup_noisyand_inference_model()
        elif inference_model_type == 'LeakyOR':
            self._setup_leakyor_inference_model()
        elif inference_model_type == 'LeakyAND':
            self._setup_leakyand_inference_model()
        elif inference_model_type == 'SimpleOR':
            self._setup_simpleor_inference_model()
        else:
            print("Default setting up the inference model to NoisyOR")
            self._setup_noisyor_inference_model()

    def _update_inference_model(self):
        if isinstance(self.inference_model, NoisyANDInferenceModel):
            inference_model_type = "NoisyAND"
        elif isinstance(self.inference_model, LeakyANDInferenceModel):
            inference_model_type = "LeakyAND"
        elif isinstance(self.inference_model, NoisyORInferenceModel):
            inference_model_type = "NoisyOR"
        elif isinstance(self.inference_model, LeakyORInferenceModel):
            inference_model_type = "LeakyOR"
        elif isinstance(self.inference_model, SimpleORInferenceModel):
            inference_model_type = "SimpleOR"
        else:
            inference_model_type = None
            print("Inference model not handled")

        self._setup_inference_model(inference_model_type)

    def _setup_params(self, params):
        assert isinstance(params, dict), "params must be a dict"
        assert all((i in list(params.keys()) for i in ('prior', 'learn', 'forget', 'slip', 'guess')))
        knowledge_components = self.get_knowledge_components()

        for param_name in ('prior', 'learn', 'forget', 'leak'):
            if param_name in params.keys():
                self.params[param_name] = {kc: params[param_name][kc.id] for kc in knowledge_components}

        for param_name in ('slip', 'guess'):
            """
            self.params[param_name] = {
                x: params[param_name][x.id] for kc in knowledge_components for x in kc.get_exercises()
            }
            """
            self.params[param_name] = {kc: params[param_name][kc.id] for kc in knowledge_components}

        for param_name in ('c', 'c0', 's', 's0'):
            self.params[param_name] = {target_key: {
                source_key: params[param_name][target_key.id][source_key.id] for source_key in
                self.set_of_prerequisite_links.get_anticausal_map()[target_key].keys()}
                for target_key in self.set_of_prerequisite_links.get_anticausal_map().keys()
            }

        if 'c' not in self.params.keys():
            self.params['c'] = {}
            self.params['c0'] = {}
            c_params = {'strong': .9, 'weak': .7}
            for target_key in self.set_of_prerequisite_links.get_anticausal_map().keys():
                self.params['c'][target_key] = {}
                self.params['c0'][target_key] = {}
                for source_key in self.set_of_prerequisite_links.get_anticausal_map()[target_key].keys():
                    self.params['c'][target_key][source_key] = c_params[
                        self.set_of_prerequisite_links.get_prerequisite_link_strength(source_key, target_key)]
                    self.params['c0'][target_key][source_key] = c_params[
                        self.set_of_prerequisite_links.get_prerequisite_link_strength(source_key, target_key)]

        if 's' not in self.params.keys():
            self.params['s'] = {}
            self.params['s0'] = {}
            s_params = {'strong': .1, 'weak': .3}
            for target_key in self.set_of_prerequisite_links.get_anticausal_map().keys():
                self.params['s'][target_key] = {}
                self.params['s0'][target_key] = {}
                for source_key in self.set_of_prerequisite_links.get_anticausal_map()[target_key].keys():
                    self.params['s'][target_key][source_key] = s_params[
                        self.set_of_prerequisite_links.get_prerequisite_link_strength(source_key, target_key)]
                    self.params['s0'][target_key][source_key] = s_params[
                        self.set_of_prerequisite_links.get_prerequisite_link_strength(source_key, target_key)]

        self._setup_inference_model(self.inference_model_type)

    def set_parameters(self, parameters):

        for param_key in ('prior', 'learn', 'forget'):
            sub_parameters = parameters[param_key]

            # convert to correct format if keys of parameters are int rather than KCs
            if all((isinstance(elt, int) for elt in sub_parameters.keys())):
                sub_parameters = {self.get_kc_from_id(elt): sub_parameters[elt] for elt in sub_parameters.keys()}

            if all((key in self.params[param_key] for key in sub_parameters.keys())):
                self.params[param_key] = sub_parameters
            else:
                self.params[param_key] = {self.get_kc_from_id(key.id): sub_parameters[key] for key in
                                          sub_parameters.keys()}

        if all(('slip' in parameters.keys(), 'guess' in parameters.keys())):
            for param_key in ('slip', 'guess'):
                sub_parameters = parameters[param_key]

                # convert to correct format if keys of parameters are int rather than KCs
                if all((isinstance(elt, int) for elt in sub_parameters.keys())):
                    sub_parameters = {self.get_kc_from_id(elt): sub_parameters[elt] for elt in sub_parameters.keys()}

                if all((key in self.params[param_key] for key in sub_parameters.keys())):
                    self.params[param_key] = sub_parameters
                else:
                    self.params[param_key] = {self.get_kc_from_id(key.id): sub_parameters[key] for key in
                                              sub_parameters.keys()}

        # c parameters
        for param_key in ('c', 'c0', 's', 's0'):
            sub_parameters = parameters[param_key]

            # convert to correct format if keys of parameters are int rather than KCs
            if all((isinstance(elt, int) for elt in sub_parameters.keys())):
                sub_parameters = {self.get_kc_from_id(elt): {
                    self.get_kc_from_id(subelt): sub_parameters[elt][subelt] for subelt in sub_parameters[elt].keys()
                } for elt in sub_parameters.keys()}

            if all((key in self.params[param_key] for key in sub_parameters.keys())):
                self.params[param_key] = sub_parameters
            else:
                self.params[param_key] = {self.get_kc_from_id(key.id): {
                    self.get_kc_from_id(subkey.id): sub_parameters[key][subkey] for subkey in sub_parameters[key]
                } for key in sub_parameters.keys()}

        self._setup_inference_model(self.inference_model_type)

    def add_learner(self, learner):
        """
        Add a learner in the LearnerPool.
        :param learner: Learner object, the learner to be added
        """
        if isinstance(learner, list):
            for elt in learner:
                print(elt)
                self.add_learner(elt)

        if all((elt is not learner for elt in self.learners)):
            if learner.id != 0:
                self.learners.append(learner)
                learner.set_learner_pool(self)

    def set_prerequisite_link_strength(self, source_kc, target_kc, strength):
        """
        Set the strength of the linked between source_kc and target_kc to given value.
        :param source_kc: the source kc of the link
        :param target_kc: the target kc of the link
        :param strength: the wished strength of the link
        """
        self.set_of_prerequisite_links.set_prerequisite_link_strength(source_kc, target_kc)

    def get_prerequisite_link_strength(self, source_kc, target_kc):
        """
        Returns the strength of a prerequisite link given its source and target.
        :param source_kc: KnowledgeComponent object, the source of the studied prerequisite link
        :param target_kc: KnowledgeComponent object, the target of the studied prerequisite link
        :return: str, the strength of the prerequisite link
        """
        return self.set_of_prerequisite_links.get_prerequisite_link_strength(source_kc, target_kc)

    def get_prerequisite_link_map(self):
        """
        Returns the map of the prerequisite links.
        :return: dict, the map of the prerequisite links
        """
        return self.set_of_prerequisite_links.get_anticausal_map()

    def get_knowledge_components(self):
        """
        Return all KCs of the domain associated to the LearnerPool.
        :return: the list of Domain's KCs.
        """
        return self.set_of_knowledge_components.get_knowledge_components()

    def get_kc_from_id(self, kc_id):
        """
        Return the corresponding KnowledgeComponent from LearnerPool SetOfKCs
        :param kc_id: the id of the wanted KC
        :return: the KC
        """
        return self.set_of_knowledge_components.get_kc_from_id(kc_id)

    def get_exercise_from_id(self, exercise_id):
        """
        Return the corresponding Exercise from LearnerPool SetOfExercises
        :param exercise_id: the id of the wanted Exercise
        :return: the Exercise
        """
        return self.set_of_exercises.get_exercise_from_id(exercise_id)

    def get_learner_ids(self):
        """
        Return the ids of the learners that belong to LearnerPool.
        :return: the list of learners' ids.
        """
        return [learner.id for learner in self.learners]

    def get_learner_from_id(self, learner_id):
        """
        Get a Learner object that corresponds to learner_id id.
        :param learner_id: id of the searched learner
        :return: Learner object that has learner_id as id
        """
        return [learner for learner in self.learners if learner.id == learner_id]

    def get_random_learner(self):
        """
        Return a random learner among the learners belonging to the LearnerPool.
        :return: Learner object, a random learner
        """
        from random import seed
        from random import randint
        # seed random number generator
        seed(1)
        return self.learners[randint(0, len(self.learners))]

    def get_prior(self, kc):
        """
        Returns the value of prior parameter of the LearnerPool for given kc
        :param kc: the kc of which we want to get the prior parameter
        :return: the value of the parameter
        """
        if isinstance(kc, int):
            kc = self.get_kc_from_id(kc_id=kc)
        return self.params['prior'][kc]

    def set_prior(self, kc, value):
        """
        Set the value of the LearnerPool's prior parameter on given KC to a given value
        :param kc: the kc we want to change the prior parameter
        :param value: the value of the new prior parameter
        """
        if isinstance(kc, int):
            kc = self.get_kc_from_id(kc_id=kc)
        self.params['prior'][kc] = value
        self._update_inference_model()

    def get_learn(self, kc):
        """
        Returns the value of learn parameter of the LearnerPool for given kc
        :param kc: the kc of which we want to get the learn parameter
        :return: the value of the learn parameter
        """
        if isinstance(kc, int):
            kc = self.get_kc_from_id(kc_id=kc)
        return self.params['learn'][kc]

    def set_learn(self, kc, value):
        if isinstance(kc, int):
            kc = self.get_kc_from_id(kc_id=kc)
        self.params['learn'][kc] = value
        self._update_inference_model()

    def get_forget(self, kc):
        """
        Returns the value of forget parameter of the LearnerPool for given kc
        :param kc: the kc of which we want to get the forget parameter
        :return: the value of the forget parameter
        """
        if isinstance(kc, int):
            kc = self.get_kc_from_id(kc_id=kc)

        return self.params['forget'][kc]

    def set_forget(self, kc, value):
        if isinstance(kc, int):
            kc = self.get_kc_from_id(kc_id=kc)
        self.params['forget'][kc] = value
        self._update_inference_model()

    '''
    def get_slip(self, exercise):
        """
        Returns the value of slip parameter of the LearnerPool for given exercise
        :param exercise: the exercise of which we want to get the slip parameter
        :return: the value of the slip parameter
        """
        if isinstance(exercise, int):
            exercise = self.get_exercise_from_id(exercise_id=exercise)
        return self.params['slip'][exercise]

    def set_slip(self, exercise, value):
        if isinstance(exercise, int):
            exercise = self.get_exercise_from_id(exercise_id=exercise)
        self.params['slip'][exercise] = value

    def get_guess(self, exercise):
        """
        Returns the value of guess parameter of the LearnerPool for given exercise
        :param exercise: the exercise of which we want to get the guess parameter
        :return: the value of the guess parameter
        """
        if isinstance(exercise, int):
            exercise = self.get_exercise_from_id(exercise_id=exercise)
        return self.params['guess'][exercise]

    def set_guess(self, exercise, value):
        if isinstance(exercise, int):
            exercise = self.get_exercise_from_id(exercise_id=exercise)
        self.params['guess'][exercise] = value
    '''

    def get_slip(self, kc):
        """
        Returns the value of slip parameter of the LearnerPool for given exercise
        :param kc: the exercise of which we want to get the slip parameter
        :return: the value of the slip parameter
        """
        if isinstance(kc, int):
            kc = self.get_kc_from_id(kc_id=kc)
        return self.params['slip'][kc]

    def set_slip(self, kc, value):
        if isinstance(kc, int):
            kc = self.get_kc_from_id(kc_id=kc)
        self.params['slip'][kc] = value

    def get_guess(self, kc):
        """
        Returns the value of guess parameter of the LearnerPool for given exercise
        :param kc: the exercise of which we want to get the guess parameter
        :return: the value of the guess parameter
        """
        if isinstance(kc, int):
            kc = self.get_kc_from_id(kc_id=kc)
        return self.params['guess'][kc]

    def set_guess(self, kc, value):
        if isinstance(kc, int):
            kc = self.get_kc_from_id(kc_id=kc)
        self.params['guess'][kc] = value

    def get_c0_param(self, source, target):
        """
        Return the c params associated to the LearnerPool.
        :return: the c params
        """
        if isinstance(source, int):
            source = self.get_kc_from_id(kc_id=source)

        if isinstance(target, int):
            target = self.get_kc_from_id(kc_id=target)

        if 'c0' in self.params.keys():
            return self.params['c0'][target][source]
        else:
            return None

    def get_c_param(self, source, target):
        """
        Return the c params associated to the LearnerPool.
        :return: the c params
        """
        if isinstance(source, int):
            source = self.get_kc_from_id(kc_id=source)

        if isinstance(target, int):
            target = self.get_kc_from_id(kc_id=target)

        if 'c' in self.params.keys():
            return self.params['c'][target][source]
        else:
            return None

    def get_c0_parameters(self):
        """
        Return the c params associated to the LearnerPool.
        :return: the c params
        """
        if 'c0' in self.params.keys():
            c_parameters = {
                target.id: {source.id: self.params['c0'][target][source] for source in self.params['c0'][target].keys()}
                for target in self.params['c0'].keys()}
            return pd.DataFrame(c_parameters)
        else:
            return None

    def get_c_parameters(self):
        """
        Return the c params associated to the LearnerPool.
        :return: the c params
        """
        if 'c' in self.params.keys():
            c_parameters = {
                target.id: {source.id: self.params['c'][target][source] for source in self.params['c'][target].keys()}
                for target in self.params['c'].keys()}
            return pd.DataFrame(c_parameters)
        else:
            return None

    def set_c0_param(self, source, target, value):
        """
        Return the c params associated to the LearnerPool.
        :return: the c params
        """
        if isinstance(source, int):
            source = self.get_kc_from_id(kc_id=source)
        if isinstance(target, int):
            target = self.get_kc_from_id(kc_id=target)
        self.params['c0'][target][source] = value
        self._update_inference_model()

    def set_c_param(self, source, target, value):
        """
        Return the c params associated to the LearnerPool.
        :return: the c params
        """
        if isinstance(source, int):
            source = self.get_kc_from_id(kc_id=source)
        if isinstance(target, int):
            target = self.get_kc_from_id(kc_id=target)
        self.params['c'][target][source] = value
        self._update_inference_model()

    def get_s0_param(self, source, target):
        """
        Return the s params associated to the LearnerPool.
        :return: the s params
        """
        if isinstance(source, int):
            source = self.get_kc_from_id(kc_id=source)

        if isinstance(target, int):
            target = self.get_kc_from_id(kc_id=target)

        if 's0' in self.params.keys():
            return self.params['s0'][target][source]
        else:
            return None

    def get_s_param(self, source, target):
        """
        Return the s params associated to the LearnerPool.
        :return: the s params
        """
        if isinstance(source, int):
            source = self.get_kc_from_id(kc_id=source)

        if isinstance(target, int):
            target = self.get_kc_from_id(kc_id=target)

        if 's' in self.params.keys():
            return self.params['s'][target][source]
        else:
            return None

    def get_s0_parameters(self):
        """
        Return the c params associated to the LearnerPool.
        :return: the c params
        """
        if 's0' in self.params.keys():
            s_parameters = {target.id: {
                source.id: self.params['s0'][target][source] for source in self.params['s0'][target].keys()
            } for target in self.params['s0'].keys()}
            return pd.DataFrame(s_parameters)
        else:
            return None

    def get_s_parameters(self):
        """
        Return the c params associated to the LearnerPool.
        :return: the c params
        """
        if 's' in self.params.keys():
            s_parameters = {target.id: {
                source.id: self.params['s'][target][source] for source in self.params['s'][target].keys()
            } for target in self.params['s'].keys()}
            return pd.DataFrame(s_parameters)
        else:
            return None

    def set_s0_param(self, source, target, value):
        """
        Return the c params associated to the LearnerPool.
        :return: the c params
        """
        if isinstance(source, int):
            source = self.get_kc_from_id(kc_id=source)
        if isinstance(target, int):
            target = self.get_kc_from_id(kc_id=target)
        self.params['s0'][target][source] = value
        self._update_inference_model()

    def set_s_param(self, source, target, value):
        """
        Return the c params associated to the LearnerPool.
        :return: the c params
        """
        if isinstance(source, int):
            source = self.get_kc_from_id(kc_id=source)
        if isinstance(target, int):
            target = self.get_kc_from_id(kc_id=target)
        self.params['s'][target][source] = value
        self._update_inference_model()

    def get_leak(self, kc):
        return self.params['leak'][kc]

    def set_leak(self, kc, value):
        if isinstance(kc, int):
            kc = self.get_kc_from_id(kc_id=kc)
        self.params['leak'][kc] = value

    def get_learner_pool_kc_children(self, kc):
        if isinstance(kc, int):
            kc = self.get_kc_from_id(kc_id=kc)
        return self.set_of_prerequisite_links.get_kc_children(kc)

    def has_kc_children(self, kc):
        if self.get_learner_pool_kc_children(kc):
            return True
        return False

    def get_learner_pool_kc_parents(self, kc):
        if isinstance(kc, int):
            kc = self.get_kc_from_id(kc_id=kc)
        return self.set_of_prerequisite_links.get_kc_parents(kc)

    def has_kc_parents(self, kc):
        if self.get_learner_pool_kc_parents(kc):
            return True
        return False

    def _setup_noisyand_inference_model(self):
        """
        Return the Noisy-AND gate inference model that corresponds to the learner pool (KCs + prerequisite links).
        :return: Noisy-AND gate inference model
        """
        self.inference_model = NoisyANDInferenceModel(self)

    def _setup_noisyor_inference_model(self):
        """
        Return the Noisy-OR gate inference model that corresponds to the learner pool (KCs + prerequisite links).
        :return: Noisy-OR gate inference model
        """
        self.inference_model = NoisyORInferenceModel(self)

    def _setup_leakyor_inference_model(self):
        """
        Return the Leaky-OR gate inference model that corresponds to the learner pool (KCs + prerequisite links).
        :return: Leaky-OR gate inference model
        """
        self.inference_model = LeakyORInferenceModel(self)

    def _setup_simpleor_inference_model(self):
        """
        Return the Leaky-OR gate inference model that corresponds to the learner pool (KCs + prerequisite links).
        :return: Leaky-OR gate inference model
        """
        self.inference_model = SimpleORInferenceModel(self)

    def _setup_leakyand_inference_model(self):
        """
        Return the Leaky-OR gate inference model that corresponds to the learner pool (KCs + prerequisite links).
        :return: Leaky-OR gate inference model
        """
        self.inference_model = LeakyANDInferenceModel(self)

    def generate_learner_traces(self, n_learners, n_transactions_min=1, n_transactions_max=10, hidden=True, verbose=2):
        """
        Generate the learner traces of n_learners with maximum n_transactions_max transactions
        :param n_learners: the number of learners of which we want to generate learner traces
        :param n_transactions_max: the maximum number of transactions a learner can have
        :return: the generated dataset
        """
        import random

        bn0, bn0_nodes = self.inference_model.get_bn0(), self.inference_model.get_bn0_nodes()
        tbn, tbn_nodes = self.inference_model.get_tbn(), self.inference_model.get_tbn_nodes()

        data = [['order_id', 'user_id', 'skill_id', 'skill_name', 'item_id', 'correct']]
        ks_lst = [["learner_id", ""]]

        idx = 0
        for i in tqdm.tqdm(range(n_learners), leave=False):
            # define how many transactions are made by learner
            n_transactions = random.randint(n_transactions_min, n_transactions_max)
            learner_traces = {}

            if hidden:
                for j in range(n_transactions):
                    selected_kc = random.choice(self.get_knowledge_components())
                    selected_item = selected_kc.get_random_exercise()

                    bn = self.inference_model.get_unrolled_dbn(j + 1)
                    ie = gum.LazyPropagation(bn)
                    ie.addTarget(f"({selected_kc.id}){j}")
                    ie.makeInference()
                    ie.setEvidence(learner_traces)
                    posterior = ie.posterior(f"({selected_kc.id}){j}").toarray()[1]

                    del ie
                    correct = 0 if random.uniform(0, 1) > posterior else 1
                    learner_traces[f"({selected_kc.id}){j}"] = correct
                    data.append([idx, i, selected_kc.id, selected_kc.name, selected_item.id, correct])
                    idx += 1

            else:
                for j in range(n_transactions):
                    if j == 0:  # learner's first transaction
                        # when it is the first transaction of the learner, we generate data from bn0 part of the dbn
                        ie = gum.LazyPropagation(bn0)
                        ie.addJointTarget(set(bn0_nodes))
                        ie.makeInference()
                        posterior = ie.jointPosterior(set(bn0_nodes))

                        values = posterior.toarray()
                        var_names = posterior.var_names

                        tt = truthtable(len(var_names))
                        combinations = [{var_names[i]: int(elt[i]) for i in range(len(elt))} for elt in tt]
                        pba_vals = np.array([[comb, values[tuple(list(comb.values()))]] for comb in combinations])
                        selected_state = random.choices(pba_vals[:, 0], weights=pba_vals[:, 1])[0]

                        ks = {n: selected_state[n] for n in bn0_nodes}

                    else:
                        # otherwise, we infer the current knowledge state knowing the state during the previous transaction
                        evs = {n: ks[n[:-1] + str(j - 1)] for n in bn0_nodes}
                        ie = gum.LazyPropagation(tbn)
                        ie.addJointTarget(set(tbn_nodes))
                        ie.setEvidence(evs)
                        ie.makeInference()
                        posterior = ie.jointPosterior(set(tbn_nodes))

                        values = posterior.toarray()
                        var_names = posterior.var_names

                        tt = truthtable(len(var_names))
                        combinations = [{var_names[i]: int(elt[i]) for i in range(len(elt))} for elt in tt]
                        pba_vals = np.array([[comb, values[tuple(list(comb.values()))]] for comb in combinations])
                        selected_state = random.choices(pba_vals[:, 0], weights=pba_vals[:, 1])[0]

                        ks = {**ks, **{n[:-1] + str(j): selected_state[n] for n in selected_state if n.endswith('t')}}

                    # random picking the evaluated resource to generate the hidden data
                    knowledge_components = self.get_knowledge_components()
                    evaluated_kc = knowledge_components[random.randint(0, len(knowledge_components) - 1)]
                    exercises = evaluated_kc.get_exercises()
                    evaluated_exercise = exercises[random.randint(0, len(exercises) - 1)]
                    data.append([
                        idx, i, evaluated_kc.id, evaluated_kc.name, evaluated_exercise.id, ks[f'({evaluated_kc.id}){j}']
                    ])
                    idx += 1
                ks_lst.append([i, ks])
                print(data)
                print(ks_lst)

        df = pd.DataFrame(data[1:], columns=data[0])
        return df

    def setup_learner_pool_parameters(self, data_df, defaults):
        from pyBKT.models import Model

        model = Model(seed=42, num_fits=1)
        model.fit(data=data_df, defaults=defaults, multigs=True, forgets=True, multilearn=True)
        params = model.params()
        print("BKT parameters are:", params)

        for kc in self.get_knowledge_components():
            try:
                self.set_prior(kc, params.loc[f'{kc.id}', 'prior', 'default'].value)
                self.set_learn(kc, params.loc[f'{kc.id}', 'learns', f'{kc.id}'].value)
                self.set_forget(kc, params.loc[f'{kc.id}', 'forgets', f'{kc.id}'].value)

                for exercise in kc.get_exercises():
                    self.set_guess(exercise, params.loc[f'{kc.id}', 'guesses', f'{exercise.id}'].value)
                    self.set_slip(exercise, params.loc[f'{kc.id}', 'slips', f'{exercise.id}'].value)
            except:
                print("Setting up the learner pool parameters with BKT has failed")

    def update_bn0_from_database(self, bn0_data):
        """
        Modify the parameters of the BN0 network according to given data
        :param bn0_data: the data on which parameters must be computed
        """
        database_columns = bn0_data.columns
        n_prior_interactions = len(bn0_data.index)

        # 1ST STEP: COMPUTE ALL PRIORS
        for col in [column for column in database_columns if not column.startswith('(Z')]:
            kc_id = int(re.search('\((.+?)\)', col).group(1))
            occ_kc_mastery = len(bn0_data[bn0_data[f'({kc_id})0'] == 1].index)
            prior = occ_kc_mastery / n_prior_interactions
            self.set_prior(kc_id, prior)
        for col in [column for column in database_columns if column.startswith('(Z')]:
            node_source_kc_id = int(re.search('\[(.+?)->', col).group(1))
            node_target_kc_id = int(re.search('->(.+?)]', col).group(1))

            if isinstance(self.inference_model, NoisyANDInferenceModel):
                sub_df_zeros = bn0_data[bn0_data[f'({node_source_kc_id})0'] == 0]
                sub_df_ones = bn0_data[bn0_data[f'({node_source_kc_id})0'] == 1]
                z1 = len(sub_df_ones[sub_df_ones[col] == 1].index)
                c = z1 / len(sub_df_ones.index)
                self.set_c0_param(source=node_source_kc_id, target=node_target_kc_id, value=c)

                z1 = len(sub_df_zeros[sub_df_zeros[col] == 1].index)
                s = z1 / len(sub_df_zeros.index)
                self.set_s0_param(source=node_source_kc_id, target=node_target_kc_id, value=s)

    def update_2tbn_from_database(self, tbn_data):
        """
        Modify the parameters of the 2TBN network according to given data
        :param tbn_data: the data on which parameters must be computed
        """
        database_columns = tbn_data.columns
        for col in [column for column in database_columns if column.startswith('(T')]:
            kc_id = int(re.search('\[(.+?)\]', col).group(1))
            if isinstance(self.inference_model, SimpleORInferenceModel):
                sub_df = tbn_data[tbn_data[f'(T[{kc_id}])t'] == 0]
                if len(sub_df.index) != 0:
                    number_of_learn_case = len(sub_df[sub_df[f'({kc_id})t'] == 1].index)
                    learn = min(number_of_learn_case / len(sub_df.index), 1-10e-8)
                    self.set_learn(kc_id, learn)
                else:
                    print(f"learn for {kc_id} leads to zero division")

                sub_df = tbn_data[tbn_data[f'(T[{kc_id}])t'] == 1]
                if len(sub_df.index) != 0:
                    number_of_forget_case = len(sub_df[sub_df[f'({kc_id})t'] == 0].index)
                    forget = max(number_of_forget_case / len(sub_df.index), 10e-8)
                    self.set_forget(kc_id, forget)
                else:
                    print(f"forget for {kc_id} leads to zero division")

                sub_df = tbn_data[tbn_data[f'({kc_id})t'] == 1]
                if len(sub_df.index) != 0:
                    for z_col in [column for column in database_columns if column.startswith(f'(Z[{kc_id}')]:
                        parent_kc_id = int(re.search('->(.+?)]', z_col).group(1))
                        ## c = # of time (Z[X->Y])t=1 when (X)t=1 divided by sum over k of # of time (Z[X->Y])t=k when (X)t=1
                        number_of_transit_case = len(sub_df[sub_df[z_col] == 1].index)
                        c = number_of_transit_case / len(sub_df.index)
                        self.set_c_param(source=parent_kc_id, target=kc_id, value=c)
                else:
                    print(f"c for {kc_id} leads to zero division")

            else:
                sub_df = tbn_data[tbn_data[f'({kc_id})0'] == 0]
                if len(sub_df.index) != 0:
                    t0, t1 = len(sub_df[sub_df[f'(T[{kc_id}])t'] == 0].index), len(
                        sub_df[sub_df[f'(T[{kc_id}])t'] == 1].index)
                    learn = min(t1 / (t0 + t1), 1 - 10e-5)
                    self.set_learn(kc_id, learn)
                else:
                    print(f"learn for {kc_id} leads to zero division")

                ## forget : # of time (T[X])t=0 when (X)0=1 divided by sum over k of # of time (T[X])t=k when (X)0=1
                sub_df = tbn_data[tbn_data[f'({kc_id})0'] == 1]
                if len(sub_df.index) != 0:
                    t0, t1 = len(sub_df[sub_df[f'(T[{kc_id}])t'] == 0].index), len(
                        sub_df[sub_df[f'(T[{kc_id}])t'] == 1].index)
                    forget = max(t0 / (t0 + t1), 10e-5)
                    self.set_forget(kc_id, forget)

                else:
                    print(f"forget for {kc_id} leads to zero division")

                # 2ND STEP: for all Y parent of X, compute the value of (X)t -> (Z[X->Y])t

                if isinstance(self.inference_model, NoisyORInferenceModel) or isinstance(self.inference_model,
                                                                                         LeakyORInferenceModel):
                    sub_df = tbn_data[tbn_data[f'({kc_id})t'] == 1]
                    if len(sub_df.index) != 0:
                        for z_col in [column for column in database_columns if column.startswith(f'(Z[{kc_id}')]:
                            parent_kc_id = int(re.search('->(.+?)]', z_col).group(1))
                            ## c = # of time (Z[X->Y])t=1 when (X)t=1 divided by sum over k of # of time (Z[X->Y])t=k when (X)t=1
                            z0, z1 = len(sub_df[sub_df[z_col] == 0].index), len(sub_df[sub_df[z_col] == 1].index)
                            c = z1 / (z0 + z1)
                            self.set_c_param(source=parent_kc_id, target=kc_id, value=c)
                    else:
                        print(f"c for {kc_id} leads to zero division")

                    if isinstance(self.inference_model, LeakyORInferenceModel):
                        # check if node has transition to it (noisyor implied it has children)
                        node_kc_children = self.get_learner_pool_kc_children(kc_id)
                        has_node_children = True if node_kc_children else False
                        if has_node_children:
                            leak = len(tbn_data[tbn_data[f'(L[{kc_id}])t'] == 1].index) / len(tbn_data.index)
                            self.set_leak(kc_id, leak)

                elif isinstance(self.inference_model, NoisyANDInferenceModel) or isinstance(self.inference_model,
                                                                                            LeakyANDInferenceModel):
                    sub_df_zeros = tbn_data[tbn_data[f'({kc_id})t'] == 0]
                    sub_df_ones = tbn_data[tbn_data[f'({kc_id})t'] == 1]

                    if len(sub_df.index) != 0:
                        for z_col in [column for column in database_columns if column.startswith(f'(Z[{kc_id}')]:
                            child_kc_id = int(re.search('->(.+?)]', z_col).group(1))
                            z1 = len(sub_df_ones[sub_df_ones[z_col] == 1].index)
                            c = z1 / len(sub_df_ones.index)
                            self.set_c_param(source=kc_id, target=child_kc_id, value=c)

                            z1 = len(sub_df_zeros[sub_df_zeros[z_col] == 1].index)
                            s = z1 / len(sub_df_zeros.index)
                            self.set_s_param(source=kc_id, target=child_kc_id, value=s)

                    if isinstance(self.inference_model, LeakyANDInferenceModel):
                        guess = len(sub_df_zeros[sub_df_zeros[f"(O[{kc_id}])t"] == 1].index) / len(sub_df_zeros.index)
                        slip = len(sub_df_ones[sub_df_ones[f"(O[{kc_id}])t"] == 0].index) / len(sub_df_ones.index)

                        self.set_slip(kc_id, slip)
                        self.set_guess(kc_id, guess)
                        """
                        # check if node has transition to it (noisyor implied it has children)
                        node_kc_parents = self.get_learner_pool_kc_parents(kc_id)
                        has_node_parents = True if node_kc_parents else False

                        if has_node_parents:
                            leak = len(tbn_data[tbn_data[f'(L[{kc_id}])t'] == 1].index) / len(tbn_data.index)
                            self.set_leak(kc_id, leak)
                        """

    def fit(self, data, fit_model, defaults, n_gibbs, burn_in_period, sample_period=1, init_type='zero',
            n_init=1, verbose=0):
        """
        Fit the parameters of the learner pool according to a given database
        :param data: the path of the csv file that contains the database or directly a DataFrame
        :param defaults: dict that indicates the correspondences between common key names and data column names
        :param n_sample
        :param n_gibbs
        :param burn_in_period
        :param sample_period
        """
        import random
        # DATA CLEANING AND INSTANTIATION
        if isinstance(data, str):
            data_df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            data_df = data
        else:
            return TypeError('data must either be str of csv file path or DataFrame')

        # TODO: data_df = data_cleaning(data_df, defaults)

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

            # TODO: give other datasets defaults (CT, ASSISTMENTS, etc)

        learners = data_df[defaults['user_id']].unique()  # TODO: add them in learner pool learners ?
        # DATA TO CONTINUOUS LEARNER TRACES
        partial_ks_list = []
        for learner in learners:  # TODO: parallelize this task
            learner_df = data_df[data_df[defaults['user_id']] == learner]
            learner_df.reset_index(drop=True, inplace=True)
            n_eval = len(learner_df.index)
            if n_eval < 2:
                continue  # TODO: add a cleaning step for the dataset

            # _1st step:_ generate n_sample rows of possible mastery state
            # each row is partial data ; for each learner, we generate n_sample dictionaries with keys
            learner_possible_partial_ks = self.get_possible_partial_knowledge_states_from_learner_interactions(
                learner_df, n_sample=1, defaults=defaults)
            partial_ks_list = partial_ks_list + learner_possible_partial_ks

            del learner_df
            del learner_possible_partial_ks

        del data_df

        prev_nll = np.inf
        nll = self.compute_negative_log_likelihood(partial_ks_list)

        if verbose == 2:
            print("Initial nll: ", nll)

        # EM PART
        t = 0
        while nll < prev_nll:
            t += 1
            if verbose == 2:
                print(f"EM step #{t}")

            prev_nll = nll
            if fit_model == 'mcem':
                nll = self._mcem_fit(partial_ks_list, n_samples=n_gibbs, verbose=verbose)
            elif fit_model == 'gibbs':
                nll = self._gibbs_fit(partial_ks_list, n_gibbs=n_gibbs, burn_in_period=burn_in_period,
                                      sample_period=sample_period, init_type=init_type, n_init=n_init, verbose=verbose)
            elif fit_model == 'blocked':
                nll = self._blocked_fit(partial_ks_list, n_gibbs=n_gibbs, burn_in_period=burn_in_period,
                                        sample_period=sample_period, verbose=verbose)
        return self

    def _gibbs_fit(self, knowledge_states, n_gibbs=100, burn_in_period=50,
                   sample_period=10, init_type='random', n_init=10, verbose=0):
        """
        Fit the parameters of the learner pool according to a given database
        :param data: the path of the csv file that contains the database or directly a DataFrame
        :param defaults: dict that indicates the correspondences between common key names and data column names
        :param n_sample
        :param n_gibbs
        :param burn_in_period
        :param sample_period
        """
        bn0_data, tbn_data = [], []
        # TODO: parallelize this task
        for knowledge_state in tqdm.tqdm(knowledge_states, disable=bool(verbose == 0), leave=False):
            samples = self.inference_model.gibbs_sampling(knowledge_state, n_gibbs=n_gibbs,
                                                          burn_in_period=burn_in_period, sample_period=sample_period,
                                                          init_type=init_type, n_init=n_init)

            bn0_data = np.concatenate((
                bn0_data,
                [{
                    key: sample[key] for key in sample.keys() if key.endswith(')0') and not key.startwith('(O')
                } for sample in samples]
            ))
            tbn_data = np.concatenate((
                tbn_data, flatten([unrolled_sample_to_2tbn_samples(sample) for sample in samples])
            ))
            del samples

        bn0_data, tbn_data = flatten(bn0_data), flatten(tbn_data)  # flattening generated data
        bn0_data_df, tbn_data_df = pd.DataFrame(bn0_data), pd.DataFrame(tbn_data)

        del bn0_data, tbn_data

        # M-STEP: SEARCHING FOR PARAMETERS CORRESPONDING TO THE COMPLETED DATABASE
        self.update_bn0_from_database(bn0_data_df)
        self.update_2tbn_from_database(tbn_data_df)
        del bn0_data_df, tbn_data_df

        if verbose == 2:
            self.print_parameters()
        # CHECK IF CONVERGENCE
        nll = self.compute_negative_log_likelihood(knowledge_states)  # TODO: replace with loglikelihood of the dbn
        if verbose == 2:
            print("likelihood", nll)
        return nll

    def _mcem_fit(self, knowledge_states, n_samples, verbose=0):
        import random
        bn0_data, tbn_data = [], []

        # TODO: parallelize this task
        for ks in tqdm.tqdm(knowledge_states, disable=bool(verbose == 0), leave=False):
            n_eval = len(ks.keys())
            bn = self.inference_model.get_unrolled_dbn(n_eval)
            nodes = self.inference_model.get_nodes_of_unrolled_dbn(n_eval)
            ie = gum.LazyPropagation(bn)
            ie.addJointTarget(set(nodes))
            if ks is not None:
                ie.setEvidence(ks)
            ie.makeInference()
            posterior = ie.jointPosterior(set(nodes))
            values = posterior.toarray().flatten()
            var_names = posterior.var_names
            samples = random.choices(range(2 ** len(var_names)), weights=values, k=n_samples)

            samples = [[int(x) for x in bin(sample)[2:]] for sample in samples]
            samples = [[0 for _ in range(len(var_names) - len(sample))] + sample for sample in samples]

            samples = [
                {var_names[i]: sample[i] for i in range(len(var_names))}
                for sample in samples
            ]

            del posterior, values, var_names, ie

            bn0_data = np.concatenate((
                bn0_data,
                [{
                    key: sample[key] for key in sample.keys() if key.endswith(')0') and not key.startswith('(O')
                } for sample in samples]
            ))

            tbn_data = np.concatenate((
                tbn_data, flatten([unrolled_sample_to_2tbn_samples(sample) for sample in samples])
            ))

            del samples
        bn0_data = flatten(bn0_data)
        tbn_data = flatten(tbn_data)

        bn0_data_df = pd.DataFrame(bn0_data)
        tbn_data_df = pd.DataFrame(tbn_data)
        del bn0_data, tbn_data
        # M-STEP: SEARCHING FOR PARAMETERS CORRESPONDING TO THE COMPLETED DATABASE
        self.update_bn0_from_database(bn0_data_df)
        self.update_2tbn_from_database(tbn_data_df)
        del bn0_data_df, tbn_data_df

        if verbose == 2:
            self.print_parameters()

        # CHECK IF CONVERGENCE
        nll = self.compute_negative_log_likelihood(knowledge_states)  # TODO: replace with loglikelihood of the dbn
        if verbose == 2:
            print("likelihood", nll)

        return nll

    def _blocked_fit(self, knowledge_states, n_gibbs, burn_in_period,
                     sample_period, init_type='random', verbose=0):
        """
        Fit the parameters of the learner pool according to a given database
        :param data: the path of the csv file that contains the database or directly a DataFrame
        :param defaults: dict that indicates the correspondences between common key names and data column names
        :param n_sample
        :param n_gibbs
        :param burn_in_period
        :param sample_period
        """
        bn0_data, tbn_data = [], []
        # TODO: parallelize this task

        for knowledge_state in tqdm.tqdm(knowledge_states, disable=bool(verbose == 0), leave=False):
            samples = self.inference_model.blocked_gibbs_sampling(knowledge_state,
                                                                  n_gibbs=n_gibbs,
                                                                  burn_in_period=burn_in_period,
                                                                  sample_period=sample_period,
                                                                  init_type=init_type)

            bn0_data = np.concatenate((
                bn0_data,
                [{
                    key: sample[key] for key in sample.keys() if key.endswith(')0') and not key.startswith('(O')
                } for sample in samples]
            ))

            tbn_data = np.concatenate((
                tbn_data, flatten([unrolled_sample_to_2tbn_samples(sample) for sample in samples])
            ))
            del samples

        bn0_data, tbn_data = flatten(bn0_data), flatten(tbn_data)  # flattening generated data
        bn0_data_df, tbn_data_df = pd.DataFrame(bn0_data), pd.DataFrame(tbn_data)

        del bn0_data, tbn_data

        # M-STEP: SEARCHING FOR PARAMETERS CORRESPONDING TO THE COMPLETED DATABASE
        self.update_bn0_from_database(bn0_data_df)
        self.update_2tbn_from_database(tbn_data_df)
        del bn0_data_df, tbn_data_df

        if verbose == 2:
            self.print_parameters()
        # CHECK IF CONVERGENCE
        nll = self.compute_negative_log_likelihood(knowledge_states)  # TODO: replace with loglikelihood of the dbn
        if verbose == 2:
            print("likelihood", nll)
        return nll

    def get_possible_partial_knowledge_states_from_learner_interactions(self, learner_interactions_df, n_sample,
                                                                        defaults):
        """
        Return the possible partial knowledge states that correspond to the interactions of the learner that belong to
        LearnerPool given a sampling rate (i.e. closest is n_sample to infinite, closest is the sampling to the real
        probability distribution).
        :param learner_interactions_df: the dataset of the learner interactions.
        :param n_sample: the number of sample to determine the probability distribution
        :param defaults: the corresponding dict of the dataset
        :return: a list of dicts that correspond to the sampled possible partial knowledge states given the learner
        interactions.
        """
        assert isinstance(learner_interactions_df, pd.DataFrame), "Learner interactions must be a DataFrame"
        possible_partial_knowledge_states = []

        if isinstance(self.inference_model, NoisyANDInferenceModel) or isinstance(
                self.inference_model, NoisyORInferenceModel):
            possible_partial_ks = {}
            for i, row in learner_interactions_df.iterrows():
                # Interaction info
                kc_id = int(row[defaults['kc_id']])
                correct = int(row[defaults['correct']])

                # For every row, we draw the mastery probability according to the learner interaction of the row
                possible_partial_ks[f'({kc_id}){i}'] = correct
            possible_partial_knowledge_states.append(possible_partial_ks)

        elif isinstance(self.inference_model, LeakyANDInferenceModel) or isinstance(
                self.inference_model, LeakyORInferenceModel):
            possible_partial_ks = {}
            for i, row in learner_interactions_df.iterrows():
                # Interaction info
                kc_id = int(row[defaults['kc_id']])
                correct = int(row[defaults['correct']])

                possible_partial_ks[f'(O[{kc_id}]){i}'] = correct

            possible_partial_knowledge_states.append(possible_partial_ks)

        return possible_partial_knowledge_states

    def compute_negative_log_likelihood(self, states):
        total = 0
        for state in states:
            bn = self.inference_model.get_unrolled_dbn(len(state.keys()))

            ie = gum.LazyPropagation(bn)

            targets = set(state.keys())
            ie.addJointTarget(targets)
            ie.makeInference()
            # posterior = ie.jointPosterior(targets).topandas()
            posterior = ie.jointPosterior(set(targets))
            values = posterior.toarray().flatten()
            var_names = posterior.var_names

            state_bin_lst = [state[name] for name in var_names]
            binary_string = ''.join(map(str, state_bin_lst))
            idx = int(binary_string, 2)
            total += np.log(values[idx])

            del posterior
            del ie
            del bn

        return -total

    def evaluate(self, data, metrics='auc', defaults="as"):

        results = []
        if isinstance(data, str):
            data_df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            data_df = data
        else:
            return TypeError('data must either be str of csv file path or DataFrame')

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

        learners = data_df[defaults['user_id']].unique()  # TODO: add them in learner pool learners ?

        exp, pred = [], []
        ks_states = []
        for learner_id in learners:  # TODO: parallelize this task
            learner_df = data_df[data_df[defaults['user_id']] == learner_id]
            learner_df.reset_index(drop=True, inplace=True)
            learner = Learner(learner_id=learner_id, learner_pool=self)

            learner_traces = [LearnerTrace(learner,
                                           exercise=self.get_exercise_from_id(int(row[defaults['exercise_id']])),
                                           success=bool(row[defaults['correct']])) for i, row in learner_df.iterrows()]

            ks_states.append(
                {f"({trace.get_kc().id}){i}": int(trace.get_success()) for i, trace in enumerate(learner_traces)})
            learner_exp = [trace.get_success() for trace in learner_traces]

            if isinstance(self.inference_model, NoisyANDInferenceModel) or isinstance(
                    self.inference_model, NoisyORInferenceModel):
                learner_pred = [self.inference_model.get_posterior_given_evidence(
                    node=f"({trace.get_kc().id}){i}",
                    evs={f"({ev_trace.get_kc().id}){ev_i}": ev_trace.get_success() for ev_i, ev_trace in
                         enumerate(learner_traces[:i])}
                ) for i, trace in enumerate(learner_traces)]

            elif isinstance(self.inference_model, LeakyANDInferenceModel) or isinstance(
                    self.inference_model, LeakyORInferenceModel):
                learner_pred = [self.inference_model.get_posterior_given_evidence(
                    node=f"(O[{trace.get_kc().id}]){i}",
                    evs={f"(O[{ev_trace.get_kc().id}]){ev_i}": ev_trace.get_success() for ev_i, ev_trace in
                         enumerate(learner_traces[:i])}
                ) for i, trace in enumerate(learner_traces)]

            else:
                break

            # predicted values

            exp = np.concatenate((exp, learner_exp))
            pred = np.concatenate((pred, learner_pred))

        if isinstance(metrics, list):
            pass

        res = self.compute_metrics(metrics, {'expected_values': exp, 'predicted_values': pred, 'ks_states': ks_states})

        return res

    def compute_metrics(self, metrics, args):
        from sklearn.metrics import roc_auc_score, mean_squared_error


        if isinstance(metrics, list):
            return {metric: self.compute_metrics(metric, args) for metric in metrics}
        else:
            if metrics == 'auc':
                auc_score = roc_auc_score(args['expected_values'], args['predicted_values'])
                return auc_score
            elif metrics == "nll":
                nll = self.compute_negative_log_likelihood(args['ks_states'])
                return nll
            elif metrics == 'rmse':
                rmse = mean_squared_error(args['expected_values'], args['predicted_values'], squared=False)
                return rmse
            """
            elif metrics == 'bic':
                n = len(args['expected_values'])
                k = 1
                bic_score = np.log(n) * k - 2 * np.log(L)
                return bic_score
            """

    def evaluate_along_time(self, data, metrics='auc', defaults="kartable"):
        from sklearn.metrics import roc_auc_score

        results = []
        if isinstance(data, str):
            data_df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            data_df = data
        else:
            return TypeError('data must either be str of csv file path or DataFrame')

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

        learners = data_df[defaults['user_id']].unique()  # TODO: add them in learner pool learners ?

        exp, pred = [], []

        n_transactions_max = max([
            len(data_df[data_df[defaults['user_id']] == learner_id].index)
            for learner_id in np.unique(data_df["user_id"])
        ])

        exps, preds = [], []

        for learner_id in learners:  # TODO: parallelize this task
            learner_df = data_df[data_df[defaults['user_id']] == learner_id]
            learner_df.reset_index(drop=True, inplace=True)
            learner = Learner(learner_id=learner_id, learner_pool=self)

            learner_traces = [LearnerTrace(learner,
                                           exercise=self.get_exercise_from_id(int(row[defaults['exercise_id']])),
                                           success=bool(row[defaults['correct']])) for i, row in learner_df.iterrows()]

            learner_exps = [1 if trace.get_success() else 0 for trace in learner_traces]

            if isinstance(self.inference_model, NoisyANDInferenceModel) or isinstance(
                    self.inference_model, NoisyORInferenceModel):
                learner_preds = [self.inference_model.get_posterior_given_evidence(
                    node=f"({trace.get_kc().id}){i}",
                    evs={f"({ev_trace.get_kc().id}){ev_i}": ev_trace.get_success() for ev_i, ev_trace in
                         enumerate(learner_traces[:i])}
                ) for i, trace in enumerate(learner_traces)]

            elif isinstance(self.inference_model, LeakyANDInferenceModel) or isinstance(
                    self.inference_model, LeakyORInferenceModel):
                learner_preds = [self.learner_pool.inference_model.get_posterior_given_evidence(
                    node=f"(O[{trace.get_kc().id}]){i}",
                    evs={f"(O[{ev_trace.get_kc().id}]){ev_i}": ev_trace.get_success() for ev_i, ev_trace in
                         enumerate(learner_traces[:i])}
                ) for i, trace in enumerate(learner_traces)]

            else:
                break

            # predicted values

            exps.append(learner_exps)
            preds.append(learner_preds)

        if isinstance(metrics, list):
            pass

        exps, preds = np.array(exps), np.array(preds)

        if metrics == 'auc':
            aucs = [roc_auc_score(exps[:, i], preds[:, i]) for i in range(n_transactions_max)]
            print('aucs along time', aucs)
        elif metrics == "nll":
            nll = self.compute_negative_log_likelihood()
            results.append(['NLL', nll])
        return results

    def setup_random_parameters(self):
        """
        Initialize LearnerPool parameters with random values
        """
        import random
        knowledge_components = self.get_knowledge_components()
        params = {
            'prior': {kc.id: random.uniform(0, .5) for kc in knowledge_components},
            'learn': {kc.id: random.uniform(0, .5) for kc in knowledge_components},
            'forget': {kc.id: random.uniform(0, .3) for kc in knowledge_components},
            'slip': {**{x.id: random.uniform(0, .25) for kc in knowledge_components for x in kc.get_exercises()},
                     **{kc.id: random.uniform(0, .25) for kc in knowledge_components}},
            'guess': {**{x.id: random.uniform(0, .25) for kc in knowledge_components for x in kc.get_exercises()},
                      **{kc.id: random.uniform(0, .25) for kc in knowledge_components}},
            'c0': {target_key.id: {
                source_key.id: random.uniform(.5, 1) for source_key in
                self.set_of_prerequisite_links.get_anticausal_map()[target_key].keys()}
                for target_key in self.set_of_prerequisite_links.get_anticausal_map().keys()
            },
            's0': {
                target_key.id: {
                    source_key.id: random.uniform(0, .5) for source_key in
                    self.set_of_prerequisite_links.get_anticausal_map()[target_key].keys()}
                for target_key in self.set_of_prerequisite_links.get_anticausal_map().keys()
            },
            'c': {target_key.id: {
                source_key.id: random.uniform(.5, 1) for source_key in
                self.set_of_prerequisite_links.get_anticausal_map()[target_key].keys()}
                for target_key in self.set_of_prerequisite_links.get_anticausal_map().keys()
            },
            's': {
                target_key.id: {
                    source_key.id: random.uniform(0, .5) for source_key in
                    self.set_of_prerequisite_links.get_anticausal_map()[target_key].keys()}
                for target_key in self.set_of_prerequisite_links.get_anticausal_map().keys()
            },
            'leak': {kc.id: random.uniform(0, .3) for kc in knowledge_components}
        }

        self._setup_params(params)

    def get_dict_params(self):
        params = {'prior': {kc.id: self.params["prior"][kc] for kc in self.get_knowledge_components()},
                  'learn': {kc.id: self.params["learn"][kc] for kc in self.get_knowledge_components()},
                  'forget': {kc.id: self.params["forget"][kc] for kc in self.get_knowledge_components()},
                  'c0': {i.id: {j.id: self.params["c0"][i][j] for j in self.params["c0"][i].keys()} for i in
                        self.params["c0"].keys()},
                  's0': {i.id: {j.id: self.params["s0"][i][j] for j in self.params["s0"][i].keys()} for i in
                        self.params["s0"].keys()},
                  'c': {i.id: {j.id: self.params["c"][i][j] for j in self.params["c"][i].keys()} for i in
                        self.params["c"].keys()},
                  's': {i.id: {j.id: self.params["s"][i][j] for j in self.params["s"][i].keys()} for i in
                        self.params["s"].keys()}}
        return params

    def print_parameters(self):
        """
        Method to print LearnerPool parameters
        """
        import pandas as pd
        kc_params_df = pd.DataFrame.from_dict(
            {kc.name: {'prior': self.params["prior"][kc],
                       'learn': self.params["learn"][kc],
                       'forget': self.params["forget"][kc]} for kc in self.get_knowledge_components()},
            orient='index')

        print("KC parameters\n:", kc_params_df)
        exercise_params_df = pd.DataFrame({key: self.params[key] for key in ['slip', 'guess']})
        print("Exercise parameters\n:", exercise_params_df)
        prerequisite_params_df = pd.DataFrame.from_dict(
            {f"{j.name}->{i.name}": {'c': self.params["c"][i][j], 's': self.params["s"][i][j]}
             for i in self.params["c"].keys()
             for j in self.params["c"][i].keys()}, orient='index')
        print("Prerequisite parameters\n:", prerequisite_params_df)

    def get_root_knowledge_components(self):
        nodes_with_parents = [key for key in self.set_of_prerequisite_links.get_anticausal_map().keys()]
        root_nodes = [kc for kc in self.get_knowledge_components() if kc not in nodes_with_parents]
        return root_nodes

    def get_prerequisite_link_topological_order(self):
        import toposort
        from itertools import chain

        graph = {}
        bn_nodes = self.domain_bn.nodes()
        for i in bn_nodes:
            graph[i] = []

        bn_arcs = self.domain_bn.arcs()
        for a, b in bn_arcs:
            graph[a].append(b)

        for key, value in graph.items():
            graph[key] = set(value)

        result = list(toposort.toposort(graph))[::-1]
        result = [list(res) for res in result]

        topological_order = list(
            chain.from_iterable([[self.domain_bn.variable(elt).name() for elt in sublist]
                                 for sublist in result])
        )
        return topological_order
