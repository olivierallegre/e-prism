import pyAgrum as gum
import pyAgrum.lib.dynamicBN as gdyn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse
import copy
import re
import random
import time
from .utils import get_graph_from_arcs, get_root_nodes, get_leaf_nodes, determine_node_type, determine_node_timestamp, \
    get_max_timestamp, truthtable
from math import prod


def showBN(bn):
    print('---------------------------------')
    for i in bn.nodes():
        print('{0} : {1}'.format(i, str(bn.variable(i))))
    print('---------------------------------')

    for (i, j) in bn.arcs():
        print('{0}->{1}'.format(bn.variable(i).name(), bn.variable(j).name()))
    print('---------------------------------')


def unroll_tbn(temp_bn, n_steps):
    dbn = gdyn.unroll2TBN(temp_bn, n_steps)
    return dbn


class InferenceModel:

    def __init__(self, learner_pool):
        """
        Init of the general inference model (doesn't depend on the gate model).
        :param learner_pool: the learner pool associated to the inference model
        """
        self.learner_pool = learner_pool
        self.bn0 = gum.BayesNet()
        self.bn0_nodes = []
        self.tbn = gum.BayesNet()
        self.tbn_nodes = []

    def get_link_strength(self, source_kc, target_kc):
        """
        Returns the strength of a prerequisite link given its source and target.
        :param source_kc: KnowledgeComponent object, the source of the studied prerequisite link
        :param target_kc: KnowledgeComponent object, the target of the studied prerequisite link
        :return: str, the strength of the prerequisite link
        """
        return self.learner_pool.get_prerequisite_link_strength(source_kc, target_kc)

    def get_link_strengths(self):
        """
        Returns all the link strengths of prerequisite links of the associated LearnerPool.
        :return: the link strengths
        """
        return self.learner_pool.get_prerequisite_link_map()

    def get_posterior_given_evidence(self, node, evs):
        # print("n ", node, "; evs ", evs.keys())
        tbn = unroll_tbn(self.tbn, len(evs) + 1)
        ie = gum.LazyPropagation(tbn)
        ie.setEvidence(evs)
        ie.makeInference()
        posterior = ie.posterior(tbn.idFromName(node)).toarray()[1]

        del tbn
        return posterior

    def get_joint_posterior(self, evs={}):
        n_transactions = len(evs.keys())
        bn = self.get_unrolled_dbn(n_transactions)
        nodes = self.get_nodes_of_unrolled_dbn(n_transactions)
        target_variables = [n for n in nodes if n not in evs.keys()]
        ie = gum.LazyPropagation(bn)
        ie.addJointTarget(set(target_variables))
        ie.setEvidence(evs)
        ie.makeInference()
        posterior = ie.jointPosterior(set(target_variables))
        values = posterior.toarray().flatten()
        var_names = posterior.var_names
        del posterior
        del ie
        return var_names, values

    def predict_knowledge_state_from_learner_traces(self, learner_traces):
        """
        Return the predicted knowledge states after given interactions with exercises (Learner traces).
        :param learner_traces: list of the LearnerTrace
        :return: dict, knowledge state in the following shape {kc: mastering probability of kc after learner traces}
        """
        knowledge_components = self.learner_pool.get_knowledge_components()
        tbn = unroll_tbn(self.tbn, len(learner_traces) + 1)
        evidence = {}

        for i, trace in enumerate(learner_traces):
            # Learner trace elements
            evaluated_kc = trace.get_kc()
            success = trace.get_success()
            exercise = trace.get_exercise()

            # Setup exercise parameters
            guess = self.learner_pool.get_guess(exercise)
            slip = self.learner_pool.get_slip(exercise)

            # Modifying the BN in function of the learner trace
            tbn.add(gum.LabelizedVariable(f"exercise({exercise.id}){i}", f"exercise({exercise.id}){i}", 2))
            tbn.addArc(f"({evaluated_kc.id}){i}", f"exercise({exercise.id}){i}")
            tbn.cpt(f"exercise({exercise.id}){i}")[{f"({evaluated_kc.id}){i}": 0}] = [1 - guess, guess]
            tbn.cpt(f"exercise({exercise.id}){i}")[{f"({evaluated_kc.id}){i}": 1}] = [slip, 1 - slip]

            # Setup the evidence
            evidence[f"exercise({exercise.id}){i}"] = int(success)

        ie = gum.LazyPropagation(tbn)
        ie.setEvidence(evidence)
        ie.makeInference()
        knowledge_states = {}
        for kc in knowledge_components:
            try:
                knowledge_states[kc] = ie.posterior(tbn.idFromName(f"({kc.id}){len(evidence.keys())}"))[1]
            except Exception as e:
                print(e)
                print(evidence)
        return knowledge_states

    def plot_knowledge_state_evolution_from_learner_traces(self, learner_traces):
        """
        Return the predicted knowledge states after given interactions with exercises (Learner traces).
        :param learner_traces: list of the LearnerTrace
        :return: dict, knowledge state in the following shape {kc: mastering probability of kc after learner traces}
        """
        knowledge_components = self.learner_pool.get_knowledge_components()
        tbn = unroll_tbn(self.tbn, len(learner_traces) + 1)
        evidence = {}

        for i, trace in enumerate(learner_traces):
            # Learner trace elements
            evaluated_kc = trace.get_kc()
            success = trace.get_success()
            exercise = trace.get_exercise()

            # Setup exercise parameters
            guess = self.learner_pool.get_guess(evaluated_kc)
            slip = self.learner_pool.get_slip(evaluated_kc)
            # Modifying the BN in function of the learner trace
            tbn.add(gum.LabelizedVariable(f"exercise({exercise.id}){i}", f"exercise({exercise.id}){i}", 2))
            tbn.addArc(f"({evaluated_kc.id}){i}", f"exercise({exercise.id}){i}")
            tbn.cpt(f"exercise({exercise.id}){i}")[{f"({evaluated_kc.id}){i}": 0}] = [1 - guess, guess]
            tbn.cpt(f"exercise({exercise.id}){i}")[{f"({evaluated_kc.id}){i}": 1}] = [slip, 1 - slip]

            # Setup the evidence
            evidence[f"exercise({exercise.id}){i}"] = int(success)

        ie = gum.LazyPropagation(tbn)
        ie.setEvidence(evidence)
        ie.makeInference()
        knowledge_states = {
            kc.id: [
                ie.posterior(tbn.idFromName(f"({kc.id}){i}"))[1] for i in range(len(evidence.keys()))
            ]
            for kc in knowledge_components
        }
        return knowledge_states

    def get_nodes_of_unrolled_dbn(self, n_steps):
        temporal_nodes = [node[:-1] for node in [n for n in self.tbn_nodes if n.endswith('t')]]

        all_nodes = self.bn0_nodes
        for i in range(1, n_steps):
            all_nodes = all_nodes + [n + str(i) for n in temporal_nodes]

        return all_nodes

    def gibbs_sampling(self, evidence, n_gibbs=1e5, burn_in_period=500, sample_period=1, gamma=1e-13, init_type='random',
                       n_init=1, verbose=0):
        import random
        result = []  # concatenation of all generated fully observed data
        n_steps = get_max_timestamp(evidence)

        n_changed_state, n_state = 0, 0
        n_changes_through_n_gibbs = []

        for _ in range(int(n_init)):
            extended_evidence = self.extend_evidence(evidence)
            state = self.initial_guess_on_hidden_nodes(extended_evidence, init_type)
            for i in range(int(n_gibbs)):
                topological_order = self.get_unrolled_topological_order(n_steps)
                hidden_nodes = [node for node in topological_order if node not in extended_evidence.keys()]
                # for every node in dbn that is not in evidence, we assess the probability of the states of the node and
                # we update it in the knowledge state

                random.shuffle(hidden_nodes)

                for node in hidden_nodes:
                    previous_state = state[node]
                    subevidence = copy.deepcopy(state)  # evidence that corresponds to the update task
                    try:
                        del subevidence[node]
                    except KeyError:
                        pass

                    # try to update directly with given rules
                    pba = self.apply_update_rules(state, subevidence, node)

                    if pba < gamma:
                        pba = gamma
                    elif abs(pba - 1) < gamma:
                        pba = 1 - gamma

                    state[node] = 1 if random.uniform(0, 1) < pba else 0

                    del subevidence

                    n_state += 1
                    if state[node] != previous_state:
                        n_changed_state += 1
                        # print("State has changed")

                if i > burn_in_period:
                    if i % sample_period == 0:
                        if self.is_state_valid(state):
                            result.append(state)

                n_changes_through_n_gibbs.append(n_changed_state / n_state)

            del state
        if verbose:  # plot the number of changes during gibbs sampling along time
            plt.figure()
            plt.plot(range(n_gibbs), n_changes_through_n_gibbs)
            plt.show()

        return result

    def check_gibbs_sampling_convergence(self, evidence, n_gibbs = 1e5, burn_in_period = 500, sample_period = 1, gamma = 1e-3, init_type = 'random',
        n_init = 1, verbose = 0):
        n_samplers = 10
        gibbs_samplers = [
            self.gibbs_sampling(evidence, n_gibbs, burn_in_period, sample_period, gamma, init_type)
            for k in range(n_samplers)
        ]
        m = len(gibbs_samplers[0])
        #fk_list = [sum([ for elt in gibbs_samplers[k]]) / m for k in range(n_samplers)]
        return 0

    def blocked_gibbs_sampling(self, evidence, n_gibbs=1000, burn_in_period=100,
                               sample_period=1, init_type="random"):
        import random
        import time

        start_time = time.time()
        result = []  # concatenation of all generated fully observed data
        n_steps = get_max_timestamp(evidence)

        extended_evidence = self.extend_evidence(evidence)
        # TODO: check if some block are entirely determined

        topological_order = self.get_unrolled_block_topological_order(n_steps)
        blocks = [block for block in topological_order]
        state = self.initial_guess_on_hidden_nodes(extended_evidence, init_type)

        for i in range(int(n_gibbs)):
            for block in blocks:
                subevidence = self.get_subevidence_from_block(block, state, extended_evidence)
                state = self.update_state_from_block_inference(state, block, subevidence)
                del subevidence
            if i > burn_in_period:
                if i % sample_period == 0:
                    if self.is_state_valid(state):
                        result.append(state)
        del state
        return result

    def get_unrolled_block_topological_order(self, n_steps):
        import itertools
        one_step_topological_order = [int(kc_id) for kc_id in self.learner_pool.get_prerequisite_link_topological_order()]
        block_topological_order = list(itertools.chain.from_iterable([
            [f"(B[{kc_id}]){i}" for kc_id in one_step_topological_order] for i in range(n_steps)
        ]))

        return block_topological_order

    def get_unrolled_topological_order(self, n_steps):
        import toposort
        from itertools import chain

        unrolled_bn = unroll_tbn(self.tbn, n_steps)
        graph = {}
        bn_nodes = unrolled_bn.nodes()
        for i in bn_nodes:
            graph[i] = []

        bn_arcs = unrolled_bn.arcs()
        for a, b in bn_arcs:
            graph[a].append(b)

        for key, value in graph.items():
            graph[key] = set(value)

        result = list(toposort.toposort(graph))[::-1]
        result = [list(res) for res in result]
        for res in result:
            random.shuffle(res)
        topological_order = list(
            chain.from_iterable([[unrolled_bn.variable(elt).name() for elt in sublist]
                                 for sublist in result])
        )
        return topological_order

    def initial_guess_on_hidden_nodes(self, evidence={}, init_type='random'):
        """
        Initialize the knowledge state with random bool for every node not in evidence
        :param nodes: the nodes of the DBN to be initialized (#TODO: delete it and use direct self computation)
        :param evidence: the state of nodes that are fixed for init
        :param init_type: str, define the type of the initialization
        :return dict, the initialized state of the DBN
        """
        # intialize parameters
        n_eval = get_max_timestamp(evidence)
        nodes = self.get_nodes_of_unrolled_dbn(n_eval)
        state = {n: evidence[n] if n in evidence.keys() else 0 for n in nodes}
        unseen_nodes = [n for n in nodes if n not in evidence.keys()]

        if init_type == 'random':
            is_initial_random_state_valid = False
            while not is_initial_random_state_valid:
                for node in unseen_nodes:
                    state[node] = random.randint(0, 1)
                is_initial_random_state_valid = self.is_state_valid(state)
        elif init_type == 'posterior':
            dbn = self.get_unrolled_dbn(n_eval)
            ie = gum.LazyPropagation(dbn)
            for node in unseen_nodes:
                ie.addJointTarget(set(np.concatenate((list(evidence.keys()), [node]))))
                ie.setEvidence(evidence)
                ie.makeInference()
                posterior = ie.posterior(node).topandas()
                pba = posterior.iloc[1]
                state[node] = 1 if random.uniform(0, 1) < pba else 0
        elif init_type == "zero":
            for node in unseen_nodes:
                state[node] = 0

        elif init_type == "one":
            for node in unseen_nodes:
                state[node] = 1

        return state

    def get_unrolled_dbn(self, n_steps):
        """
        Return the unrolled dynamic bayesian network without test items.
        :param n_steps: int, the number of steps of the DBN
        :return: the unrolled DBN
        """
        dbn = gdyn.unroll2TBN(self.tbn, n_steps)
        return dbn

    def get_unrolled_dbn_with_observables(self, observables):
        """
        Return the unrolled dynamic bayesian network that corresponds to a set of test items.
        :param observables: dict, in shape of {timestamp: exercise object}
        :return: the unrolled DBN
        """
        assert isinstance(observables, dict), "must be a dict"
        dbn = gdyn.unroll2TBN(self.tbn, len(observables.keys()))

        for timestamp in observables.keys():
            exercise = observables[timestamp]
            kc = exercise.get_kc()
            dbn.add(
                gum.LabelizedVariable(f"(E{exercise.id}){timestamp}", f"(E{exercise.id}){timestamp}", 2))
            dbn.addArc(f"({kc.id}){timestamp}", f"(E{exercise.id}){timestamp}")

            slip, guess = self.learner_pool.get_slip(exercise), self.learner_pool.get_guess(exercise)
            dbn.cpt(f"(E{exercise.id}){timestamp}")[{f"({kc.id}){timestamp}": 0}] = [1 - guess, guess]
            dbn.cpt(f"(E{exercise.id}){timestamp}")[{f"({kc.id}){timestamp}": 1}] = [slip, 1 - slip]
        return dbn

    def get_markov_blanket(self, node):
        """
        Returns the Markov blanket of a given node in the unrolled dynamic bayesian network.
        """
        timestamp = determine_node_timestamp(node)
        unrolled_bn = unroll_tbn(self.tbn, timestamp + 1)
        markov_blanket_nodes = [unrolled_bn.variable(node_id).name() for node_id in
                                gum.MarkovBlanket(unrolled_bn, node).nodes()]
        del unrolled_bn
        return markov_blanket_nodes

    def get_bn0(self):
        """
        Returns the initial bayesian network of the dbn.
        """
        return self.bn0

    def get_bn0_nodes(self):
        """
        Returns the nodes of the intiial bayesian network of the dbn
        """
        return self.bn0_nodes

    def get_tbn(self):
        """
        Returns the transition bayesian network of the dbn.
        """
        return self.tbn

    def get_tbn_nodes(self):
        """
        Returns the nodes of the transition bayesian network of the dbn
        """
        return self.tbn_nodes


class ClassicDBNInferenceModel(InferenceModel):

    def __init__(self):
        InferenceModel.__init__(self, learner_pool)


class NoisyANDInferenceModel(InferenceModel):

    def __init__(self, learner_pool):
        """
        Inherited class from InferenceModel. Corresponds to the model with Noisy-AND gates.
        :param learner_pool: the learner pool associated to the inference model
        """
        InferenceModel.__init__(self, learner_pool)
        self.setup_noisyand_dbn()

    def setup_noisyand_dbn(self):
        """
        Add corresponding nodes to the BN associated to the Noisy-AND gate inference model.
        """
        knowledge_components = self.learner_pool.get_knowledge_components()
        priors = {kc: self.learner_pool.get_prior(kc) for kc in knowledge_components}
        learns = {kc: self.learner_pool.get_learn(kc) for kc in knowledge_components}
        forgets = {kc: self.learner_pool.get_forget(kc) for kc in knowledge_components}

        # Introduce the structure of the temporal relationships between same KC's nodes
        for kc in knowledge_components:
            # for all knowledge component X, there is a node (X)0...
            if self.learner_pool.has_kc_parents(kc):
                self.bn0.addAND(gum.LabelizedVariable(f"({kc.id})0", f"({kc.id})0", 2))
            else:
                self.bn0.add(gum.LabelizedVariable(f"({kc.id})0", f"({kc.id})0", 2))
                self.bn0.cpt(f"({kc.id})0").fillWith([1 - priors[kc], priors[kc]])
            self.bn0_nodes.append(f"({kc.id})0")

            if self.learner_pool.has_kc_parents(kc):
                self.tbn.addAND(gum.LabelizedVariable(f"({kc.id})0", f"({kc.id})0", 2))
            else:
                self.tbn.add(gum.LabelizedVariable(f"({kc.id})0", f"({kc.id})0", 2))
                self.tbn.cpt(f"({kc.id})0").fillWith([1 - priors[kc], priors[kc]])
            self.tbn_nodes.append(f"({kc.id})0")

            # ... a node (T[X])t linked to (X)0...
            self.tbn.add(gum.LabelizedVariable(f"(T[{kc.id}])t", f"(T[{kc.id}])t", 2))
            self.tbn.addArc(f"({kc.id})0", f"(T[{kc.id}])t")
            self.tbn_nodes.append(f"(T[{kc.id}])t")

            # ... and an AND node (X)t, at least linked to (T[X])t
            self.tbn.addAND(gum.LabelizedVariable(f"({kc.id})t", f"({kc.id})t", 2))
            self.tbn.addArc(f"(T[{kc.id}])t", f"({kc.id})t")
            self.tbn_nodes.append(f"({kc.id})t")
            self.tbn.cpt(f"(T[{kc.id}])t")[{f"({kc.id})0": 0}] = [1 - learns[kc], learns[kc]]
            self.tbn.cpt(f"(T[{kc.id}])t")[{f"({kc.id})0": 1}] = [forgets[kc], 1 - forgets[kc]]

        for kc in knowledge_components:
            parents = self.learner_pool.get_learner_pool_kc_parents(kc)
            if parents:
                for parent in parents:
                    c0, s0 = self.learner_pool.get_c0_param(parent, kc), self.learner_pool.get_s0_param(parent, kc)
                    c, s = self.learner_pool.get_c_param(parent, kc), self.learner_pool.get_s_param(parent, kc)

                    # for all prerequisite link between kcs X and Y, there is a node (Z[X->Y])0...
                    self.bn0.add(
                        gum.LabelizedVariable(f"(Z[{parent.id}->{kc.id}])0", f"(Z[{parent.id}->{kc.id}])0", 2))
                    self.bn0_nodes.append(f"(Z[{parent.id}->{kc.id}])0")
                    self.bn0.addArc(f"({parent.id})0", f"(Z[{parent.id}->{kc.id}])0")
                    self.bn0.cpt(f"(Z[{parent.id}->{kc.id}])0")[{f"({parent.id})0": 0}] = [1 - s0, s0]
                    self.bn0.cpt(f"(Z[{parent.id}->{kc.id}])0")[{f"({parent.id})0": 1}] = [1 - c0, c0]
                    self.bn0.addArc(f"(Z[{parent.id}->{kc.id}])0", f"({kc.id})0")

                    self.tbn.add(
                        gum.LabelizedVariable(f"(Z[{parent.id}->{kc.id}])0", f"(Z[{parent.id}->{kc.id}])0", 2))
                    self.tbn_nodes.append(f"(Z[{parent.id}->{kc.id}])0")
                    self.tbn.addArc(f"({parent.id})0", f"(Z[{parent.id}->{kc.id}])0")
                    self.tbn.cpt(f"(Z[{parent.id}->{kc.id}])0")[{f"({parent.id})0": 0}] = [1 - s0, s0]
                    self.tbn.cpt(f"(Z[{parent.id}->{kc.id}])0")[{f"({parent.id})0": 1}] = [1 - c0, c0]
                    self.tbn.addArc(f"(Z[{parent.id}->{kc.id}])0", f"({kc.id})0")

                    # ... and a node (Z[X->Y])t
                    self.tbn.add(
                        gum.LabelizedVariable(f"(Z[{parent.id}->{kc.id}])t", f"(Z[{parent.id}->{kc.id}])t", 2))
                    self.tbn_nodes.append(f"(Z[{parent.id}->{kc.id}])t")
                    self.tbn.addArc(f"({parent.id})t", f"(Z[{parent.id}->{kc.id}])t")
                    self.tbn.cpt(f"(Z[{parent.id}->{kc.id}])t")[{f"({parent.id})t": 0}] = [1 - s, s]
                    self.tbn.cpt(f"(Z[{parent.id}->{kc.id}])t")[{f"({parent.id})t": 1}] = [1 - c, c]
                    self.tbn.addArc(f"(Z[{parent.id}->{kc.id}])t", f"({kc.id})t")

    def apply_update_rules(self, state, evidence, node):
        node_type = determine_node_type(node_name=node)
        node_timestamp = determine_node_timestamp(node_name=node)
        pba = None

        if node_timestamp == 0:
            if node_type == 'prerequisite':
                node_source_kc_id = int(re.search('\[(.+?)->', node).group(1))
                node_target_kc_id = int(re.search('->(.+?)]', node).group(1))

                node_target_parents = self.learner_pool.get_learner_pool_kc_parents(node_target_kc_id)
                node_target_dbn_parents = [
                    f"(Z[{parent.id}->{node_target_kc_id}])0" for parent in node_target_parents
                    if parent.id != node_source_kc_id
                ]

                if all((evidence[elt] == 1 for elt in node_target_dbn_parents)):
                    pba = evidence[f"({node_target_kc_id})0"]
                else:
                    if evidence[f"({node_source_kc_id})0"] == 1:
                        pba = self.learner_pool.get_c0_param(node_source_kc_id, node_target_kc_id)
                    else:
                        pba = self.learner_pool.get_s0_param(node_source_kc_id, node_target_kc_id)

            elif node_type == 'mastery':
                node_kc_id = int(re.search('\((.+?)\)', node).group(1))
                kc_children = self.learner_pool.get_learner_pool_kc_children(node_kc_id)
                kc_parents = self.learner_pool.get_learner_pool_kc_parents(node_kc_id)
                prior = self.learner_pool.get_prior(node_kc_id)
                learn, forget = self.learner_pool.get_learn(node_kc_id), self.learner_pool.get_forget(node_kc_id)

                # no prereq with kc x
                if not kc_children and not kc_parents:
                    if evidence[f"(T[{node_kc_id}])1"] == 1:
                        pba = prior * (1 - forget) / (prior * (1 - forget) + (1 - prior) * learn)
                    else:
                        pba = prior * forget / (prior * forget + (1 - prior) * (1 - learn))

                # kc x is a root of the prerequisite structure
                elif not kc_parents:
                    c_prod = prod([
                        self.learner_pool.get_c0_param(node_kc_id, child.id) if evidence[
                            f"(Z[{node_kc_id}->{child.id}])0"] == 1 else 1 - self.learner_pool.get_c0_param(
                            node_kc_id, child.id) for child in kc_children
                    ])
                    s_prod = prod([
                        self.learner_pool.get_s0_param(node_kc_id, child.id) if evidence[
                            f"(Z[{node_kc_id}->{child.id}])0"] == 1 else 1 - self.learner_pool.get_s0_param(
                            node_kc_id, child.id) for child in kc_children
                    ])

                    if evidence[f'(T[{node_kc_id}])1'] == 1:
                        pba = (1 - forget) * prior * c_prod / ((1 - forget) * prior * c_prod + learn * (1 - prior) * s_prod)
                    else:
                        pba = forget * prior * c_prod / (forget * prior * c_prod + (1 - learn) * (1 - prior) * s_prod)

                else:
                    nodes = list(state.keys())
                    node_parent_nodes = [n for n in nodes if n.endswith(f"{node_kc_id}])0")]

                    if all((evidence[n] == 1 for n in node_parent_nodes)):
                        pba = 1
                    else:
                        pba = 0

                """
                elif not kc_children:
                    nodes = list(state.keys())
                    node_parent_nodes = [n for n in nodes if n.endswith(f"{node_kc_id}]){node_timestamp}")]

                    if any((1 - evidence[n] for n in node_parent_nodes)):
                        pba = 0
                    elif all((evidence[n] for n in node_parent_nodes)):
                        pba = 1
                else:
                    nodes = list(state.keys())
                    node_parent_nodes = [n for n in nodes if n.endswith(f"{node_kc_id}]){node_timestamp}")]
                    kc_children = self.learner_pool.get_learner_pool_kc_children(node_kc_id)
                    if any((1 - evidence[n] for n in node_parent_nodes)):
                        pba = 0
                    else:
                        c_prod = prod([
                            self.learner_pool.get_c_param(node_kc_id, child.id) if evidence[
                                f"(Z[{node_kc_id}->{child.id}])1"] == 1 else 1 - self.learner_pool.get_c_param(
                                node_kc_id, child.id) for child in kc_children
                        ])
                        s_prod = prod([
                            self.learner_pool.get_s_param(node_kc_id, child.id) if evidence[
                                f"(Z[{node_kc_id}->{child.id}])1"] == 1 else 1 - self.learner_pool.get_s_param(
                                node_kc_id, child.id) for child in kc_children
                        ])
                        if evidence[f'(T[{node_kc_id}])1'] == 1:
                            pba = (1 - forget) * prior * c_prod / (
                                        (1 - forget) * prior * c_prod + learn * (1 - prior) * s_prod)
                        else:
                            pba = forget * prior * c_prod / (
                                        forget * prior * c_prod + (1 - learn) * (1 - prior) * s_prod)
                """
        else:
            if node_type == 'prerequisite':
                node_source_kc_id = int(re.search('\[(.+?)->', node).group(1))
                node_target_kc_id = int(re.search('->(.+?)]', node).group(1))

                # Parents of node_target in the DBN are transition nodes from target's parents + the transition node to
                # target node
                node_target_parents = self.learner_pool.get_learner_pool_kc_parents(node_target_kc_id)
                node_target_dbn_parents = [f"(T[{node_target_kc_id}]){node_timestamp}"] + [
                    f"(Z[{parent.id}->{node_target_kc_id}]){node_timestamp}" for parent in node_target_parents
                    if parent.id != node_source_kc_id]

                if all((evidence[elt] == 1 for elt in node_target_dbn_parents)):
                    pba = evidence[f"({node_target_kc_id}){node_timestamp}"]
                else:
                    if evidence[f"({node_source_kc_id}){node_timestamp}"] == 1:
                        pba = self.learner_pool.get_c_param(node_source_kc_id, node_target_kc_id)
                    else:
                        pba = self.learner_pool.get_s_param(node_source_kc_id, node_target_kc_id)

            elif node_type == 'transition':
                node_kc_id = int(re.search('\[(.+?)\]', node).group(1))
                node_kc_parents = self.learner_pool.get_learner_pool_kc_parents(node_kc_id)
                has_node_parents = True if node_kc_parents else False
                target_node_parents = [f"(Z[{parent.id}->{node_kc_id}]){node_timestamp}" for parent in node_kc_parents]

                if not has_node_parents:
                    pba = evidence[f"({node_kc_id}){node_timestamp}"]
                else:
                    if all((evidence[n] == 1 for n in target_node_parents)):
                        pba = evidence[f"({node_kc_id}){node_timestamp}"]
                    else:
                        if evidence[f"({node_kc_id}){node_timestamp - 1}"] == 0:
                            pba = self.learner_pool.get_learn(node_kc_id)
                        else:
                            pba = 1 - self.learner_pool.get_forget(node_kc_id)

            else:
                node_kc_id = int(re.search('\((.+?)\)', node).group(1))
                kc_parents = self.learner_pool.get_learner_pool_kc_parents(node_kc_id)

                if not kc_parents:
                    pba = evidence[f"(T[{node_kc_id}]){node_timestamp}"]
                else:
                    nodes = list(state.keys())
                    node_parent_nodes = [n for n in nodes if n.endswith(f"{node_kc_id}]){node_timestamp}")]

                    if all((evidence[n] == 1 for n in node_parent_nodes)):
                        pba = 1
                    else:
                        pba = 0

        return pba

    def extend_evidence(self, evs):
        n_steps = get_max_timestamp(evs)
        nodes = self.get_nodes_of_unrolled_dbn(n_steps)
        evidence = {}
        for key, value in evs.items():
            evidence[key] = value

            kc_id = int(re.search('\((.+?)\)', key).group(1))
            timestamp = determine_node_timestamp(key)
            parent_nodes = [n for n in nodes if n.endswith(f"{kc_id}]){timestamp}")]

            if value == 1:
                for node in parent_nodes:
                    evidence[node] = 1
                if timestamp > 0:
                    evidence[f"(T[{kc_id}]){timestamp}"] = 1

            elif not parent_nodes:
                if timestamp > 0:
                    evidence[f"(T[{kc_id}]){timestamp}"] = 0

            elif len(parent_nodes) == 1 and timestamp == 0:
                evidence[parent_nodes[0]] = 0

        return evidence

    def is_state_valid(self, state):
        nodes = list(state.keys())

        for node in [n for n in nodes if determine_node_type(n) == 'mastery']:
            kc_id = int(re.search('\((.+?)\)', node).group(1))
            timestamp = determine_node_timestamp(node)
            parent_nodes = [n for n in nodes if n.endswith(f"{kc_id}]){timestamp}")]

            if parent_nodes:
                if state[node] == 1 and any((1 - state[n] for n in parent_nodes)):
                    return False
                elif state[node] == 0 and all((state[n] for n in parent_nodes)):
                    return False

        return True

    def get_subevidence_from_block(self, block_name, full_state, evidence):
        n_eval = get_max_timestamp(evidence) - 1
        t = determine_node_timestamp(block_name)
        block_id = int(re.search('B\[(.+?)\]', block_name).group(1))

        if t == 0:
            outer_nodes = [f"({parent.id})0" for parent in self.learner_pool.get_learner_pool_kc_parents(block_id)] + [
                f"(T[{block_id}])1"
            ]
        elif t == n_eval:
            outer_nodes = [
                f"({parent.id}){t}" for parent in self.learner_pool.get_learner_pool_kc_parents(block_id)] + [
                f"({block_id}){t-1}"] + [
                f"(Z[{block_id}->{child.id}]){t}" for child in self.learner_pool.get_learner_pool_kc_children(block_id)
            ]
        else:
            outer_nodes = [
                f"({parent.id}){t}" for parent in self.learner_pool.get_learner_pool_kc_parents(block_id)] + [
                f"({block_id}){t-1}", f"(T[{block_id}]){t+1}"] + [
                f"(Z[{block_id}->{child.id}]){t}" for child in self.learner_pool.get_learner_pool_kc_children(block_id)
            ]

        subevidence = {**{n: evidence[n] for n in evidence.keys()},
                       **{n: full_state[n] for n in outer_nodes if n not in evidence.keys()}
                       }

        return subevidence

    def update_state_from_block_inference(self, previous_state, block_name, evidence):
        # on créé le block à partir de son nom
        n_eval = get_max_timestamp(evidence) - 1
        t = determine_node_timestamp(block_name)
        block_id = int(re.search('B\[(.+?)\]', block_name).group(1))
        if t == 0:  # si c'est t=0
            bn = self.get_initial_block_bn(block_id, evidence)
        elif t == n_eval:  # si c'est t=n_eval
            bn = self.get_final_block_bn(block_id, t, evidence)
        else:  # sinon
            bn = self.get_other_block_bn(block_id, t, evidence)
        nodes = bn.names()
        inner_nodes = {node for node in nodes if node not in evidence.keys()}
        block_evidence = {n: evidence[n] for n in nodes if n in evidence.keys()}

        if all((n in block_evidence for n in nodes)):
            state = previous_state
            del bn
        else:
            ie = gum.LazyPropagation(bn)
            ie.addJointTarget(nodes)
            ie.setEvidence(block_evidence)
            try:
                ie.makeInference()
                posterior = ie.jointPosterior(inner_nodes)
            except:
                print(nodes, block_evidence, inner_nodes)
                for i in bn.nodes():
                    print(bn.cpt(i))
            values = posterior.toarray().flatten()
            var_names = posterior.var_names
            block_state_bin = random.choices(range(len(values)), weights=values)[0]
            block_state = {var_names[j]: elt for j, elt in enumerate(list(bin(block_state_bin)[2:].zfill(len(var_names))))}
            """
            tt = truthtable(len(var_names))
            combinations = [{var_names[i]: int(elt[i]) for i in range(len(elt))} for elt in tt]
            block_pba_values = np.array([[comb, values[tuple(list(comb.values()))]] for comb in combinations])
            block_state = random.choices(block_pba_values[:, 0], weights=block_pba_values[:, 1])[0]
            """

            state = {n: int(block_state[n]) if n in block_state.keys() else previous_state[n] for n in previous_state.keys()}
            del posterior, ie, bn
        return state

    def get_initial_block_bn(self, block_id, evidence):
        bn = gum.BayesNet()
        block_kc = self.learner_pool.get_kc_from_id(block_id)

        prior = self.learner_pool.get_prior(block_kc)
        learn = self.learner_pool.get_learn(block_kc)
        forget = self.learner_pool.get_forget(block_kc)

        # Introduce the structure of the temporal relationships between same KC's nodes
        # for all knowledge component X, there is a node (X)0...
        if self.learner_pool.has_kc_parents(block_kc):
            bn.addAND(gum.LabelizedVariable(f"({block_kc.id})0", f"({block_kc.id})0", 2))
        else:
            bn.add(gum.LabelizedVariable(f"({block_kc.id})0", f"({block_kc.id})0", 2))
            bn.cpt(f"({block_kc.id})0").fillWith([1 - prior, prior])

        # ... a node (T[X])t linked to (X)0...
        bn.add(gum.LabelizedVariable(f"(T[{block_kc.id}])1", f"(T[{block_kc.id}])1", 2))
        bn.addArc(f"({block_kc.id})0", f"(T[{block_kc.id}])1")
        bn.cpt(f"(T[{block_kc.id}])1")[{f"({block_kc.id})0": 0}] = [1 - learn, learn]
        bn.cpt(f"(T[{block_kc.id}])1")[{f"({block_kc.id})0": 1}] = [forget, 1 - forget]

        parents = self.learner_pool.get_learner_pool_kc_parents(block_kc)
        if parents:
            for parent in parents:
                c0, s0 = self.learner_pool.get_c0_param(parent, block_kc), self.learner_pool.get_s0_param(parent, block_kc)
                bn.add(gum.LabelizedVariable(f"({parent.id})0", f"({parent.id})0", 2))
                bn.cpt(f"({parent.id})0").fillWith([1 - evidence[f"({parent.id})0"], evidence[f"({parent.id})0"]])
                bn.add(
                    gum.LabelizedVariable(f"(Z[{parent.id}->{block_kc.id}])0", f"(Z[{parent.id}->{block_kc.id}])0", 2))
                bn.addArc(f"({parent.id})0", f"(Z[{parent.id}->{block_kc.id}])0")
                bn.cpt(f"(Z[{parent.id}->{block_kc.id}])0")[{f"({parent.id})0": 0}] = [1 - s0, s0]
                bn.cpt(f"(Z[{parent.id}->{block_kc.id}])0")[{f"({parent.id})0": 1}] = [1 - c0, c0]
                bn.addArc(f"(Z[{parent.id}->{block_kc.id}])0", f"({block_kc.id})0")

        children = self.learner_pool.get_learner_pool_kc_children(block_kc)
        if children:
            for child in children:
                c0, s0 = self.learner_pool.get_c0_param(block_kc, child), self.learner_pool.get_s0_param(block_kc, child)
                bn.add(
                    gum.LabelizedVariable(f"(Z[{block_kc.id}->{child.id}])0", f"(Z[{block_kc.id}->{child.id}])0", 2))
                bn.addArc(f"({block_kc.id})0", f"(Z[{block_kc.id}->{child.id}])0")
                bn.cpt(f"(Z[{block_kc.id}->{child.id}])0")[{f"({block_kc.id})0": 0}] = [1 - s0, s0]
                bn.cpt(f"(Z[{block_kc.id}->{child.id}])0")[{f"({block_kc.id})0": 1}] = [1 - c0, c0]

        return bn

    def get_final_block_bn(self, block_id, n_final, evidence):
        bn = gum.BayesNet()
        block_kc = self.learner_pool.get_kc_from_id(block_id)

        learn = self.learner_pool.get_learn(block_kc)
        forget = self.learner_pool.get_forget(block_kc)

        # Introduce the structure of the temporal relationships between same KC's nodes
        # for all knowledge component X, there is a node (X)0...
        bn.addAND(gum.LabelizedVariable(f"({block_kc.id}){n_final}", f"({block_kc.id}){n_final}", 2))

        # ... a node (T[X])t linked to (X)0...
        bn.add(gum.LabelizedVariable(f"({block_kc.id}){n_final-1}", f"({block_kc.id}){n_final-1}", 2))
        bn.cpt(f"({block_kc.id}){n_final - 1}").fillWith([1 - evidence[f"({block_kc.id}){n_final - 1}"],
                                                            evidence[f"({block_kc.id}){n_final - 1}"]])

        bn.add(gum.LabelizedVariable(f"(T[{block_kc.id}]){n_final}", f"(T[{block_kc.id}]){n_final}", 2))
        bn.addArc(f"(T[{block_kc.id}]){n_final}", f"({block_kc.id}){n_final}")
        bn.addArc(f"({block_kc.id}){n_final-1}", f"(T[{block_kc.id}]){n_final}")
        bn.cpt(f"(T[{block_kc.id}]){n_final}")[{f"({block_kc.id}){n_final-1}": 0}] = [1 - learn, learn]
        bn.cpt(f"(T[{block_kc.id}]){n_final}")[{f"({block_kc.id}){n_final-1}": 1}] = [forget, 1 - forget]


        parents = self.learner_pool.get_learner_pool_kc_parents(block_kc)
        if parents:
            for parent in parents:
                c, s = self.learner_pool.get_c_param(parent, block_kc), self.learner_pool.get_s_param(parent, block_kc)
                bn.add(gum.LabelizedVariable(f"({parent.id}){n_final}", f"({parent.id}){n_final}", 2))
                bn.cpt(f"({parent.id}){n_final}").fillWith([1 - evidence[f"({parent.id}){n_final}"],
                                                            evidence[f"({parent.id}){n_final}"]])
                bn.add(
                    gum.LabelizedVariable(
                        f"(Z[{parent.id}->{block_kc.id}]){n_final}", f"(Z[{parent.id}->{block_kc.id}]){n_final}", 2))
                bn.addArc(f"({parent.id}){n_final}", f"(Z[{parent.id}->{block_kc.id}]){n_final}")
                bn.cpt(f"(Z[{parent.id}->{block_kc.id}]){n_final}")[{f"({parent.id}){n_final}": 0}] = [1 - s, s]
                bn.cpt(f"(Z[{parent.id}->{block_kc.id}]){n_final}")[{f"({parent.id}){n_final}": 1}] = [1 - c, c]
                bn.addArc(f"(Z[{parent.id}->{block_kc.id}]){n_final}", f"({block_kc.id}){n_final}")

        children = self.learner_pool.get_learner_pool_kc_children(block_kc)
        if children:
            for child in children:
                c, s = self.learner_pool.get_c_param(block_kc, child), self.learner_pool.get_s_param(block_kc, child)
                bn.add(
                    gum.LabelizedVariable(
                        f"(Z[{block_kc.id}->{child.id}]){n_final}", f"(Z[{block_kc.id}->{child.id}]){n_final}", 2))
                bn.addArc(f"({block_kc.id}){n_final}", f"(Z[{block_kc.id}->{child.id}]){n_final}")
                bn.cpt(f"(Z[{block_kc.id}->{child.id}]){n_final}")[{f"({block_kc.id}){n_final}": 0}] = [1 - s, s]
                bn.cpt(f"(Z[{block_kc.id}->{child.id}]){n_final}")[{f"({block_kc.id}){n_final}": 1}] = [1 - c, c]

        return bn

    def get_other_block_bn(self, block_id, timestamp, evidence):
        bn = gum.BayesNet()
        block_kc = self.learner_pool.get_kc_from_id(block_id)
        learn = self.learner_pool.get_learn(block_kc)
        forget = self.learner_pool.get_forget(block_kc)
        # Introduce the structure of the temporal relationships between same KC's nodes
        # for all knowledge component X, there is a node (X)0...
        bn.addAND(gum.LabelizedVariable(f"({block_kc.id}){timestamp}", f"({block_kc.id}){timestamp}", 2))

        # ... a node (T[X])t linked to (X)0...
        bn.add(gum.LabelizedVariable(f"({block_kc.id}){timestamp - 1}", f"({block_kc.id}){timestamp - 1}", 2))

        bn.cpt(f"({block_kc.id}){timestamp - 1}").fillWith([1 - evidence[f"({block_kc.id}){timestamp - 1}"],
                                                            evidence[f"({block_kc.id}){timestamp - 1}"]])

        bn.add(gum.LabelizedVariable(f"(T[{block_kc.id}]){timestamp}", f"(T[{block_kc.id}]){timestamp}", 2))
        bn.add(gum.LabelizedVariable(f"(T[{block_kc.id}]){timestamp+1}", f"(T[{block_kc.id}]){timestamp+1}", 2))

        bn.addArc(f"({block_kc.id}){timestamp-1}", f"(T[{block_kc.id}]){timestamp}")
        bn.addArc(f"(T[{block_kc.id}]){timestamp}", f"({block_kc.id}){timestamp}")
        bn.addArc(f"({block_kc.id}){timestamp}", f"(T[{block_kc.id}]){timestamp + 1}")

        bn.cpt(f"(T[{block_kc.id}]){timestamp+1}")[{f"({block_kc.id}){timestamp}": 0}] = [1 - learn, learn]
        bn.cpt(f"(T[{block_kc.id}]){timestamp+1}")[{f"({block_kc.id}){timestamp}": 1}] = [forget, 1 - forget]

        bn.cpt(f"(T[{block_kc.id}]){timestamp}")[{f"({block_kc.id}){timestamp - 1}": 0}] = [1 - learn, learn]
        bn.cpt(f"(T[{block_kc.id}]){timestamp}")[{f"({block_kc.id}){timestamp - 1}": 1}] = [forget, 1 - forget]
        parents = self.learner_pool.get_learner_pool_kc_parents(block_kc)
        if parents:
            for parent in parents:
                c, s = self.learner_pool.get_c_param(parent, block_kc), self.learner_pool.get_s_param(parent, block_kc)
                bn.add(gum.LabelizedVariable(f"({parent.id}){timestamp}", f"({parent.id}){timestamp}", 2))
                bn.cpt(f"({parent.id}){timestamp}").fillWith([1 - evidence[f"({parent.id}){timestamp}"],
                                                              evidence[f"({parent.id}){timestamp}"]])
                bn.add(
                    gum.LabelizedVariable(
                        f"(Z[{parent.id}->{block_kc.id}]){timestamp}", f"(Z[{parent.id}->{block_kc.id}]){timestamp}", 2))
                bn.addArc(f"({parent.id}){timestamp}", f"(Z[{parent.id}->{block_kc.id}]){timestamp}")
                bn.cpt(f"(Z[{parent.id}->{block_kc.id}]){timestamp}")[{f"({parent.id}){timestamp}": 0}] = [1 - s, s]
                bn.cpt(f"(Z[{parent.id}->{block_kc.id}]){timestamp}")[{f"({parent.id}){timestamp}": 1}] = [1 - c, c]
                bn.addArc(f"(Z[{parent.id}->{block_kc.id}]){timestamp}", f"({block_kc.id}){timestamp}")
        children = self.learner_pool.get_learner_pool_kc_children(block_kc)
        if children:
            for child in children:
                c, s = self.learner_pool.get_c_param(block_kc, child), self.learner_pool.get_s_param(block_kc, child)
                bn.add(
                    gum.LabelizedVariable(
                        f"(Z[{block_kc.id}->{child.id}]){timestamp}", f"(Z[{block_kc.id}->{child.id}]){timestamp}", 2))
                bn.addArc(f"({block_kc.id}){timestamp}", f"(Z[{block_kc.id}->{child.id}]){timestamp}")
                bn.cpt(f"(Z[{block_kc.id}->{child.id}]){timestamp}")[{f"({block_kc.id}){timestamp}": 0}] = [1 - s, s]
                bn.cpt(f"(Z[{block_kc.id}->{child.id}]){timestamp}")[{f"({block_kc.id}){timestamp}": 1}] = [1 - c, c]

        return bn


class NoisyORInferenceModel(InferenceModel):

    def __init__(self, learner_pool):
        """
        Inherited class from InferenceModel. Corresponds to the model with Noisy-OR gates.
        :param learner_pool: the learner pool associated to the inference model
        """
        InferenceModel.__init__(self, learner_pool)
        self.setup_noisyor_dbn()

    def setup_noisyor_dbn(self):
        """
        Add corresponding nodes to the BN associated to the Noisy-OR gate inference model.
        """
        knowledge_components = self.learner_pool.get_knowledge_components()
        priors = {kc: self.learner_pool.get_prior(kc) for kc in knowledge_components}
        learns = {kc: self.learner_pool.get_learn(kc) for kc in knowledge_components}
        forgets = {kc: self.learner_pool.get_forget(kc) for kc in knowledge_components}

        # Introduce the structure of the temporal relationships between same KC's nodes
        for kc in knowledge_components:
            # for all knowledge component X, there is a node (X)0...
            if self.learner_pool.has_kc_children(kc):
                self.bn0.addOR(gum.LabelizedVariable(f"({kc.id})0", f"({kc.id})0", 2))
            else:
                self.bn0.add(gum.LabelizedVariable(f"({kc.id})0", f"({kc.id})0", 2))
                self.bn0.cpt(f"({kc.id})0").fillWith([1 - priors[kc], priors[kc]])
            self.bn0_nodes.append(f"({kc.id})0")

            if self.learner_pool.has_kc_children(kc):
                self.tbn.addOR(gum.LabelizedVariable(f"({kc.id})0", f"({kc.id})0", 2))
            else:
                self.tbn.add(gum.LabelizedVariable(f"({kc.id})0", f"({kc.id})0", 2))
                self.tbn.cpt(f"({kc.id})0").fillWith([1 - priors[kc], priors[kc]])
            self.tbn_nodes.append(f"({kc.id})0")

            # ... a node (T[X])t linked to (X)0...
            self.tbn.add(gum.LabelizedVariable(f"(T[{kc.id}])t", f"(T[{kc.id}])t", 2))
            self.tbn.addArc(f"({kc.id})0", f"(T[{kc.id}])t")
            self.tbn_nodes.append(f"(T[{kc.id}])t")

            # ... and a OR node (X)t, at least linked to (T[X])t
            self.tbn.addOR(gum.LabelizedVariable(f"({kc.id})t", f"({kc.id})t", 2))
            self.tbn.addArc(f"(T[{kc.id}])t", f"({kc.id})t")
            self.tbn_nodes.append(f"({kc.id})t")
            self.tbn.cpt(f"(T[{kc.id}])t")[{f"({kc.id})0": 0}] = [1 - learns[kc], learns[kc]]
            self.tbn.cpt(f"(T[{kc.id}])t")[{f"({kc.id})0": 1}] = [forgets[kc], 1 - forgets[kc]]

        for kc in knowledge_components:
            children = self.learner_pool.get_learner_pool_kc_children(kc)
            if children:
                for child in children:
                    c = self.learner_pool.get_c_param(kc, child)
                    # for all prerequisite link between kcs X and Y, there is a node (Z[X->Y])0...
                    self.bn0.add(
                        gum.LabelizedVariable(f"(Z[{child.id}->{kc.id}])0", f"(Z[{child.id}->{kc.id}])0", 2))
                    self.bn0_nodes.append(f"(Z[{child.id}->{kc.id}])0")
                    self.bn0.addArc(f"({child.id})0", f"(Z[{child.id}->{kc.id}])0")
                    self.bn0.cpt(f"(Z[{child.id}->{kc.id}])0")[{f"({child.id})0": 0}] = [1, 0]
                    self.bn0.cpt(f"(Z[{child.id}->{kc.id}])0")[{f"({child.id})0": 1}] = [1 - c, c]
                    self.bn0.addArc(f"(Z[{child.id}->{kc.id}])0", f"({kc.id})0")

                    # for all prerequisite link between kcs X and Y, there is a node (Z[X->Y])0...
                    self.tbn.add(
                        gum.LabelizedVariable(f"(Z[{child.id}->{kc.id}])0", f"(Z[{child.id}->{kc.id}])0", 2))
                    self.tbn_nodes.append(f"(Z[{child.id}->{kc.id}])0")
                    self.tbn.addArc(f"({child.id})0", f"(Z[{child.id}->{kc.id}])0")
                    self.tbn.cpt(f"(Z[{child.id}->{kc.id}])0")[{f"({child.id})0": 0}] = [1, 0]
                    self.tbn.cpt(f"(Z[{child.id}->{kc.id}])0")[{f"({child.id})0": 1}] = [1 - c, c]
                    self.tbn.addArc(f"(Z[{child.id}->{kc.id}])0", f"({kc.id})0")

                    # ... and a node (Z[X->Y])t
                    self.tbn.add(
                        gum.LabelizedVariable(f"(Z[{child.id}->{kc.id}])t", f"(Z[{child.id}->{kc.id}])t", 2))
                    self.tbn_nodes.append(f"(Z[{child.id}->{kc.id}])t")
                    self.tbn.addArc(f"({child.id})t", f"(Z[{child.id}->{kc.id}])t")
                    self.tbn.cpt(f"(Z[{child.id}->{kc.id}])t")[{f"({child.id})t": 0}] = [1, 0]
                    self.tbn.cpt(f"(Z[{child.id}->{kc.id}])t")[{f"({child.id})t": 1}] = [1 - c, c]
                    self.tbn.addArc(f"(Z[{child.id}->{kc.id}])t", f"({kc.id})t")

    def apply_update_rules(self, state, evidence, node):

        node_type = determine_node_type(node_name=node)
        node_timestamp = determine_node_timestamp(node_name=node)
        pba = None

        if node_timestamp == 0:
            nodes = list(state.keys())
            if node_type == 'mastery':
                node_kc_id = int(re.search('\((.+?)\)', node).group(1))
                kc_children = self.learner_pool.get_learner_pool_kc_children(node_kc_id)
                kc_parents = self.learner_pool.get_learner_pool_kc_parents(node_kc_id)
                if not kc_children and not kc_parents:
                    prior = self.learner_pool.get_prior(node_kc_id)
                    learn, forget = self.learner_pool.get_learn(node_kc_id), self.learner_pool.get_forget(node_kc_id)
                    if evidence[f"(T[{node_kc_id}])1"] == 1:
                        pba = prior * (1 - forget) / (prior * (1 - forget) + (1 - prior) * learn)
                    else:
                        pba = prior * forget / (prior * forget + (1 - prior) * (1 - learn))
                elif not kc_children:
                    node_child_nodes = [n for n in nodes if
                                        n.startswith(f"(Z[{node_kc_id}") and n.endswith(f"){node_timestamp}")]

                    if any((evidence[n] for n in node_child_nodes)):
                        pba = 1

                    else:
                        learn, forget = self.learner_pool.get_learn(node_kc_id), self.learner_pool.get_forget(node_kc_id)

                        p_plus = prod([
                            1 - self.learner_pool.get_c_param(
                                int(re.search('->(.+?)\]', child_node).group(1)),
                                int(re.search('\[(.+?)->', child_node).group(1))
                            ) for child_node in node_child_nodes])
                        p_plus = p_plus * forget if evidence[f"(T[{node_kc_id}])1"] == 1 else p_plus * (1 - forget)
                        p_minus = learn if evidence[f"(T[{node_kc_id}])1"] == 1 else 1 - learn
                        prior = self.learner_pool.get_prior(node_kc_id)
                        pba = prior * p_plus / (prior * p_plus + (1-prior) * p_minus)
                elif not kc_parents:
                    node_parent_nodes = [n for n in nodes if n.endswith(f"{node_kc_id}])0")]

                    if any((evidence[n] for n in node_parent_nodes)):
                        pba = 1
                    else:
                        pba = 0

                else:
                    node_parent_nodes = [n for n in nodes if n.endswith(f"{node_kc_id}])0")]
                    node_child_nodes = [n for n in nodes if
                                        n.startswith(f"(Z[{node_kc_id}") and n.endswith(f"){node_timestamp}")]

                    if any((evidence[n] for n in node_parent_nodes)):
                        if any((evidence[n] for n in node_child_nodes)):
                            pba = 1

                        else:
                            learn, forget = self.learner_pool.get_learn(node_kc_id), self.learner_pool.get_forget(
                                node_kc_id)
                            p_plus = prod([
                                1 - self.learner_pool.get_c_param(
                                    int(re.search('->(.+?)\]', child_node).group(1)),
                                    int(re.search('\[(.+?)->', child_node).group(1))
                                ) for child_node in node_child_nodes])
                            p_plus = p_plus * forget if evidence[f"(T[{node_kc_id}])1"] == 1 else p_plus * (1 - forget)
                            p_minus = learn if evidence[f"(T[{node_kc_id}])1"] == 1 else 1 - learn
                            prior = self.learner_pool.get_prior(node_kc_id)
                            pba = prior * p_plus / (prior * p_plus + (1 - prior) * p_minus)

                    else:
                        pba = 0

            else:
                node_source_kc_id = int(re.search('\[(.+?)->', node).group(1))
                node_target_kc_id = int(re.search('->(.+?)]', node).group(1))

                # Parents of node_target in the DBN are transition nodes from target's children + the transition node to target node
                node_target_children = self.learner_pool.get_learner_pool_kc_children(node_target_kc_id)
                node_target_dbn_parents = [
                    f"(Z[{child.id}->{node_target_kc_id}]){node_timestamp}" for child in node_target_children
                    if child.id != node_source_kc_id]

                if evidence[f"({node_source_kc_id}){node_timestamp}"] == 0:
                    pba = 0
                elif evidence[f"({node_target_kc_id}){node_timestamp}"] == 0:
                    pba = 0
                elif all((evidence[elt] == 0 for elt in node_target_dbn_parents)):
                    pba = 1
                else:
                    pba = self.learner_pool.get_c_param(node_target_kc_id, node_source_kc_id)
        else:
            if node_type == 'prerequisite':
                node_source_kc_id = int(re.search('\[(.+?)->', node).group(1))
                node_target_kc_id = int(re.search('->(.+?)]', node).group(1))

                # Parents of node_target in the DBN are transition nodes from target's children + the transition node to target node
                node_target_children = self.learner_pool.get_learner_pool_kc_children(node_target_kc_id)
                node_target_dbn_parents = [f"(T[{node_target_kc_id}]){node_timestamp}"] + [
                    f"(Z[{child.id}->{node_target_kc_id}]){node_timestamp}" for child in node_target_children
                    if child.id != node_source_kc_id]

                if evidence[f"({node_target_kc_id}){node_timestamp}"] == 0:
                    pba = 0
                elif evidence[f"({node_source_kc_id}){node_timestamp}"] == 0:
                    pba = 0
                elif all((evidence[elt] == 0 for elt in node_target_dbn_parents)):
                    pba = 1
                else:
                    pba = self.learner_pool.get_c_param(node_target_kc_id, node_source_kc_id)

            elif node_type == 'transition':
                node_kc_id = int(re.search('\[(.+?)\]', node).group(1))
                node_kc_children = self.learner_pool.get_learner_pool_kc_children(node_kc_id)
                if evidence[f"({node_kc_id}){node_timestamp}"] == 0:
                    pba = 0
                elif not node_kc_children:
                    pba = 1
                elif all((evidence[f"(Z[{child.id}->{node_kc_id}]){node_timestamp}"] == 0 for child in node_kc_children)):
                    pba = 1
                elif evidence[f"({node_kc_id}){node_timestamp - 1}"] == 0:
                    pba = self.learner_pool.get_learn(node_kc_id)
                elif evidence[f"({node_kc_id}){node_timestamp - 1}"] == 1:
                    pba = 1 - self.learner_pool.get_forget(node_kc_id)
            else:
                node_kc_id = int(re.search('\((.+?)\)', node).group(1))
                nodes = list(state.keys())
                node_parent_nodes = [n for n in nodes if n.endswith(f"{node_kc_id}]){node_timestamp}")]

                if any((evidence[n] for n in node_parent_nodes)):
                    pba = 1
                else:
                    pba = 0

        return pba

    def extend_evidence(self, evs, nodes):
        evidence = {}
        for key, value in evs.items():
            evidence[key] = value

            kc_id = int(re.search('\((.+?)\)', key).group(1))
            timestamp = determine_node_timestamp(key)
            parent_nodes = [n for n in nodes if n.startswith("(Z[") and n.endswith(f"{kc_id}]){timestamp}")]
            child_nodes = [n for n in nodes if
                           n.startswith(f"(Z[{kc_id}") and n.endswith(f"){timestamp}")]

            if value == 0:
                for node in parent_nodes + child_nodes:
                    evidence[node] = 0
                    evidence[f"(T[{kc_id}]){timestamp}"] = 0
            elif not parent_nodes:
                if timestamp > 0:
                    evidence[f"(T[{kc_id}]){timestamp}"] = 1

        return evidence

    def is_state_valid(self, state):
        nodes = list(state.keys())

        for node in [n for n in nodes if determine_node_type(n) == 'mastery']:
            kc_id = int(re.search('\((.+?)\)', node).group(1))
            timestamp = determine_node_timestamp(node)
            parent_nodes = [n for n in nodes if n.endswith(f"{kc_id}]){timestamp}")]

            if parent_nodes:
                if state[node] == 1 and all((1 - state[n] for n in parent_nodes)):
                    return False
                elif state[node] == 0 and any((state[n] for n in parent_nodes)):
                    return False

        return True


class SimpleORInferenceModel(InferenceModel):

    def __init__(self, learner_pool):
        """
        Inherited class from InferenceModel. Corresponds to the model with Noisy-OR gates.
        :param learner_pool: the learner pool associated to the inference model
        """
        InferenceModel.__init__(self, learner_pool)
        self.setup_simpleor_dbn()

    def setup_simpleor_dbn(self):
        """
        Add corresponding nodes to the BN associated to the Noisy-OR gate inference model.
        """
        knowledge_components = self.learner_pool.get_knowledge_components()
        priors = {kc: self.learner_pool.get_prior(kc) for kc in knowledge_components}
        learns = {kc: self.learner_pool.get_learn(kc) for kc in knowledge_components}
        forgets = {kc: self.learner_pool.get_forget(kc) for kc in knowledge_components}
        # Introduce the structure of the temporal relationships between same KC's nodes
        for kc in knowledge_components:
            # Introduce node for KC at time 0
            self.bn.add(gum.LabelizedVariable(f"({kc.id})0", f"({kc.id})0", 2))
            self.bn.cpt(f"({kc.id})0").fillWith([1 - priors[kc], priors[kc]])
            self.nodes.append(f"({kc.id})0")
            self.bn.addOR(gum.LabelizedVariable(f"(T[{kc.id}])t", f"(T[{kc.id}])t", 2))
            self.nodes.append(f"(T[{kc.id}])t")
            self.bn.addArc(f"({kc.id})0", f"(T[{kc.id}])t")

            self.bn.add(gum.LabelizedVariable(f"({kc.id})t", f"({kc.id})t", 2))
            self.nodes.append(f"({kc.id})t")
            self.bn.addArc(f"(T[{kc.id}])t", f"({kc.id})t")

            self.bn.cpt(f"({kc.id})t")[{f"(T[{kc.id}])t": 0}] = [1 - learns[kc], learns[kc]]
            self.bn.cpt(f"({kc.id})t")[{f"(T[{kc.id}])t": 1}] = [forgets[kc], 1 - forgets[kc]]

        for kc in knowledge_components:
            children = self.learner_pool.get_learner_pool_kc_children(kc)
            if children:
                for child in children:
                    c = self.learner_pool.get_c_param(kc, child)
                    self.bn.add(gum.LabelizedVariable(f"(Z[{child.id}->{kc.id}])t", f"(Z[{child.id}->{kc.id}])t", 2))
                    self.nodes.append(f"(Z[{child.id}->{kc.id}])t")

                    self.bn.addArc(f"({child.id})0", f"(Z[{child.id}->{kc.id}])t")
                    self.bn.addArc(f"(Z[{child.id}->{kc.id}])t", f"(T[{kc.id}])t")

                    self.bn.cpt(f"(Z[{child.id}->{kc.id}])t")[{f"({child.id})0": 0}] = [1, 0]
                    self.bn.cpt(f"(Z[{child.id}->{kc.id}])t")[{f"({child.id})0": 1}] = [1 - c, c]

    def apply_update_rules(self, state, evidence, node):

        node_type = determine_node_type(node_name=node)
        node_timestamp = determine_node_timestamp(node_name=node)
        pba = None

        if node_type == "transition":
            node_kc_id = int(re.search('\[(.+?)\]', node).group(1))
            node_kc_children = self.learner_pool.get_learner_pool_kc_children(node_kc_id)
            has_node_children = True if node_kc_children else False

            if not has_node_children:
                pba = int(evidence[f"({node_kc_id}){node_timestamp - 1}"])
            elif all((evidence[f"(Z[{child.id}->{node_kc_id}]){node_timestamp}"] == 0 for child in node_kc_children)):
                pba = int(evidence[f"({node_kc_id}){node_timestamp - 1}"])
            else:
                pba = 1

        elif node_type == 'prerequisite':
            node_source_kc_id = int(re.search('\[(.+?)->', node).group(1))
            node_target_kc_id = int(re.search('->(.+?)]', node).group(1))

            node_target_children = self.learner_pool.get_learner_pool_kc_children(node_target_kc_id)
            node_target_dbn_parents = [f"({node_target_kc_id}){node_timestamp - 1}"] + [
                f"(Z[{child.id}->{node_target_kc_id}]){node_timestamp}" for child in node_target_children
                if child.id != node_source_kc_id]

            if evidence[f"({node_source_kc_id}){node_timestamp - 1}"] == 0:
                pba = 0
            elif evidence[f"(T[{node_target_kc_id}]){node_timestamp}"] == 0:
                pba = 0
            elif all((evidence[elt] == 0 for elt in node_target_dbn_parents)):
                pba = 1

        elif node_type == 'mastery':  # node type mastery
            node_kc_id = int(re.search('\((.+?)\)', node).group(1))
            node_kc_children = self.learner_pool.get_learner_pool_kc_children(node_kc_id)
            has_node_children = True if node_kc_children else False

            if f"(T[{node_kc_id}]){node_timestamp + 1}" in evidence.keys():
                if not has_node_children:
                    pba = int(evidence[f"(T[{node_kc_id}]){node_timestamp + 1}"])
                elif evidence[f"(T[{node_kc_id}]){node_timestamp + 1}"] == 0:
                    pba = 0
                elif all((evidence[f"(Z[{child.id}->{node_kc_id}]){node_timestamp + 1}"] == 0 for child in
                          node_kc_children)):
                    pba = int(evidence[f"(T[{node_kc_id}]){node_timestamp + 1}"])

        return pba


class LeakyORInferenceModel(InferenceModel):

    def __init__(self, learner_pool):
        """
        Inherited class from InferenceModel. Corresponds to the model with Noisy-OR gates.
        :param learner_pool: the learner pool associated to the inference model
        """
        InferenceModel.__init__(self, learner_pool)
        self.setup_leakyor_dbn()

    def setup_leakyor_dbn(self):
        """
        Add corresponding nodes to the BN associated to the Noisy-OR gate inference model.
        """
        knowledge_components = self.learner_pool.get_knowledge_components()
        priors = {kc: self.learner_pool.get_prior(kc) for kc in knowledge_components}
        learns = {kc: self.learner_pool.get_learn(kc) for kc in knowledge_components}
        forgets = {kc: self.learner_pool.get_forget(kc) for kc in knowledge_components}
        leaks = {kc: self.learner_pool.get_leak(kc) for kc in knowledge_components}
        link_strengths = self.get_link_strengths()
        # Introduce the structure of the temporal relationships between same KC's nodes
        leaf_nodes = get_leaf_nodes(knowledge_components, link_strengths)
        for kc in knowledge_components:
            # Introduce node for KC at time 0
            self.bn.add(gum.LabelizedVariable(f"({kc.id})0", f"({kc.id})0", 2))
            self.bn.cpt(f"({kc.id})0").fillWith([1 - priors[kc], priors[kc]])
            """
            if kc in leaf_nodes:
                self.bn.add(gum.LabelizedVariable(f"({kc.id})0", f"({kc.id})0", 2))
                self.bn.cpt(f"({kc.id})0").fillWith([1 - priors[kc], priors[kc]])
            else:
                self.bn.addOR(gum.LabelizedVariable(f"({kc.id})0", f"({kc.id})0", 2))
            """
            self.nodes.append(f"({kc.id})0")
            self.bn.add(gum.LabelizedVariable(f"(T[{kc.id}])t", f"(T[{kc.id}])t", 2))
            self.nodes.append(f"(T[{kc.id}])t")
            self.bn.addArc(f"({kc.id})0", f"(T[{kc.id}])t")

            self.bn.addOR(gum.LabelizedVariable(f"({kc.id})t", f"({kc.id})t", 2))
            self.nodes.append(f"({kc.id})t")
            self.bn.addArc(f"(T[{kc.id}])t", f"({kc.id})t")

            self.bn.cpt(f"(T[{kc.id}])t")[{f"({kc.id})0": 0}] = [1 - learns[kc], learns[kc]]
            self.bn.cpt(f"(T[{kc.id}])t")[{f"({kc.id})0": 1}] = [forgets[kc], 1 - forgets[kc]]

        for kc in knowledge_components:
            children = self.learner_pool.get_learner_pool_kc_children(kc)

            if children:
                self.bn.add(
                    gum.LabelizedVariable(f"(L[{kc.id}])t", f"(L[{kc.id}])t", 2))
                self.bn.cpt(f"(L[{kc.id}])t").fillWith([1 - leaks[kc], leaks[kc]])
                self.nodes.append(f"(L[{kc.id}])t")
                self.bn.addArc(f"(L[{kc.id}])t", f"({kc.id})t")

                for child in children:
                    self.bn.add(
                        gum.LabelizedVariable(f"(Z[{child.id}->{kc.id}])t", f"(Z[{child.id}->{kc.id}])t", 2))

                    self.nodes.append(f"(Z[{child.id}->{kc.id}])t")

                    c = self.learner_pool.get_c_param(kc, child)
                    self.bn.addArc(f"({child.id})t", f"(Z[{child.id}->{kc.id}])t")

                    self.bn.cpt(f"(Z[{child.id}->{kc.id}])t")[{f"({child.id})t": 0}] = [1, 0]
                    self.bn.cpt(f"(Z[{child.id}->{kc.id}])t")[{f"({child.id})t": 1}] = [1 - c, c]

                    self.bn.addArc(f"(Z[{child.id}->{kc.id}])t", f"({kc.id})t")

    def apply_update_rules(self, state, evidence, node):

        node_type = determine_node_type(node_name=node)
        node_timestamp = determine_node_timestamp(node_name=node)
        pba = None

        if node_type == 'prerequisite':
            node_source_kc_id = int(re.search('\[(.+?)->', node).group(1))
            node_target_kc_id = int(re.search('->(.+?)]', node).group(1))

            # Parents of node_target in the DBN are transition nodes from target's children + the transition node to target node
            node_target_children = self.learner_pool.get_learner_pool_kc_children(node_target_kc_id)
            node_target_dbn_parents = [f"(T[{node_target_kc_id}]){node_timestamp}"] + [
                f"(L[{node_target_kc_id}]){node_timestamp}"] + [
                                          f"(Z[{child.id}->{node_target_kc_id}]){node_timestamp}" for child in
                                          node_target_children
                                          if child.id != node_source_kc_id]

            if evidence[f"({node_target_kc_id}){node_timestamp}"] == 0:
                pba = 0
            elif all((evidence[elt] == 0 for elt in node_target_dbn_parents)):
                pba = 1
            elif evidence[f"({node_source_kc_id}){node_timestamp}"] == 0:
                pba = 0

        elif node_type == 'transition':
            node_kc_id = int(re.search('\[(.+?)\]', node).group(1))
            node_kc_children = self.learner_pool.get_learner_pool_kc_children(node_kc_id)
            has_node_children = True if node_kc_children else False

            if has_node_children:
                node_dbn_parents = [f"(L[{node_kc_id}]){node_timestamp}"] + [
                    f"(Z[{child.id}->{node_kc_id}]){node_timestamp}" for child in node_kc_children]
            else:
                node_dbn_parents = None
            if evidence[f"({node_kc_id}){node_timestamp}"] == 0:
                pba = 0
            elif not has_node_children:
                pba = 1
            elif all((evidence[dbn_parent] == 0 for dbn_parent in node_dbn_parents)):
                pba = 1

        elif node_type == 'leak':
            node_kc_id = int(re.search('\[(.+?)\]', node).group(1))
            node_children = self.learner_pool.get_learner_pool_kc_children(node_kc_id)
            node_dbn_parents = [f"(T[{node_kc_id}]){node_timestamp}"] + [
                f"(Z[{child.id}->{node_kc_id}]){node_timestamp}" for child in node_children
            ]

            if evidence[f"({node_kc_id}){node_timestamp}"] == 0:
                pba = 0
            elif all((evidence[elt] == 0 for elt in node_dbn_parents)):
                pba = 1

        else:
            node_kc_id = int(re.search('\((.+?)\)', node).group(1))
            nodes = list(state.keys())
            node_parent_nodes = [n for n in nodes if n.endswith(f"{node_kc_id}]){node_timestamp}")]
            node_child_nodes = [n for n in nodes if
                                n.startswith(f"(Z[{node_kc_id}") and n.endswith(f"){node_timestamp}")]

            if any((evidence[n] for n in node_parent_nodes)):
                pba = 1
            elif all((1 - evidence[n] for n in node_parent_nodes)):
                pba = 0
            elif any((evidence[n] for n in node_child_nodes)):
                pba = 1
        return pba


class LeakyANDInferenceModel(InferenceModel):

    def __init__(self, learner_pool):
        """
        Inherited class from InferenceModel. Corresponds to the model with Noisy-AND gates.
        :param learner_pool: the learner pool associated to the inference model
        """
        InferenceModel.__init__(self, learner_pool)
        self.setup_leakyand_dbn()

    def setup_leakyand_dbn(self):
        """
        Add corresponding nodes to the BN associated to the Noisy-AND gate inference model.
        """
        knowledge_components = self.learner_pool.get_knowledge_components()
        priors = {kc: self.learner_pool.get_prior(kc) for kc in knowledge_components}
        learns = {kc: self.learner_pool.get_learn(kc) for kc in knowledge_components}
        forgets = {kc: self.learner_pool.get_forget(kc) for kc in knowledge_components}

        slips = {kc: self.learner_pool.get_slip(kc) for kc in knowledge_components}
        guesses = {kc: self.learner_pool.get_guess(kc) for kc in knowledge_components}

        # Introduce the structure of the temporal relationships between same KC's nodes
        for kc in knowledge_components:
            # creating the BN0 network
            # for all knowledge component X, there is a node (X)0...
            if self.learner_pool.has_kc_parents(kc):
                self.bn0.addAND(gum.LabelizedVariable(f"({kc.id})0", f"({kc.id})0", 2))
            else:
                self.bn0.add(gum.LabelizedVariable(f"({kc.id})0", f"({kc.id})0", 2))
                self.bn0.cpt(f"({kc.id})0").fillWith([1 - priors[kc], priors[kc]])
            self.bn0_nodes.append(f"({kc.id})0")

            # observational node (O[kc])0
            self.bn0.add(gum.LabelizedVariable(f"(O[{kc.id}])0", f"(O[{kc.id}])0", 2))
            self.bn0.addArc(f"({kc.id})0", f"(O[{kc.id}])0")
            self.bn0.cpt(f"(O[{kc.id}])0")[{f"({kc.id})0": 0}] = [1 - guesses[kc], guesses[kc]]
            self.bn0.cpt(f"(O[{kc.id}])0")[{f"({kc.id})0": 1}] = [slips[kc], 1 - slips[kc]]
            self.bn0_nodes.append(f"(O[{kc.id}])0")

            # create the 2TBN network
            if self.learner_pool.has_kc_parents(kc):
                self.tbn.addAND(gum.LabelizedVariable(f"({kc.id})0", f"({kc.id})0", 2))
            else:
                self.tbn.add(gum.LabelizedVariable(f"({kc.id})0", f"({kc.id})0", 2))
                self.tbn.cpt(f"({kc.id})0").fillWith([1 - priors[kc], priors[kc]])
            self.tbn_nodes.append(f"({kc.id})0")

            # observational node (O[kc])0
            self.tbn.add(gum.LabelizedVariable(f"(O[{kc.id}])0", f"(O[{kc.id}])0", 2))
            self.tbn.addArc(f"({kc.id})0", f"(O[{kc.id}])0")
            self.tbn.cpt(f"(O[{kc.id}])0")[{f"({kc.id})0": 0}] = [1 - guesses[kc], guesses[kc]]
            self.tbn.cpt(f"(O[{kc.id}])0")[{f"({kc.id})0": 1}] = [slips[kc], 1 - slips[kc]]
            self.tbn_nodes.append(f"(O[{kc.id}])0")

            # ... a node (T[X])t linked to (X)0...
            self.tbn.add(gum.LabelizedVariable(f"(T[{kc.id}])t", f"(T[{kc.id}])t", 2))
            self.tbn.addArc(f"({kc.id})0", f"(T[{kc.id}])t")
            self.tbn_nodes.append(f"(T[{kc.id}])t")

            # ... and an AND node (X)t, at least linked to (T[X])t
            self.tbn.addAND(gum.LabelizedVariable(f"({kc.id})t", f"({kc.id})t", 2))
            self.tbn.addArc(f"(T[{kc.id}])t", f"({kc.id})t")
            self.tbn_nodes.append(f"({kc.id})t")
            self.tbn.cpt(f"(T[{kc.id}])t")[{f"({kc.id})0": 0}] = [1 - learns[kc], learns[kc]]
            self.tbn.cpt(f"(T[{kc.id}])t")[{f"({kc.id})0": 1}] = [forgets[kc], 1 - forgets[kc]]

            self.tbn.add(gum.LabelizedVariable(f"(O[{kc.id}])t", f"(O[{kc.id}])t", 2))
            self.tbn.addArc(f"({kc.id})t", f"(O[{kc.id}])t")
            self.tbn_nodes.append(f"(O[{kc.id}])t")
            self.tbn.cpt(f"(O[{kc.id}])t")[{f"({kc.id})t": 0}] = [1 - guesses[kc], guesses[kc]]
            self.tbn.cpt(f"(O[{kc.id}])t")[{f"({kc.id})t": 1}] = [slips[kc], 1 - slips[kc]]

        for kc in knowledge_components:
            parents = self.learner_pool.get_learner_pool_kc_parents(kc)
            if parents:
                for parent in parents:
                    c, s = self.learner_pool.get_c_param(parent, kc), self.learner_pool.get_s_param(parent, kc)

                    # for all prerequisite link between kcs X and Y, there is a node (Z[X->Y])0...
                    self.bn0.add(
                        gum.LabelizedVariable(f"(Z[{parent.id}->{kc.id}])0", f"(Z[{parent.id}->{kc.id}])0", 2))
                    self.bn0_nodes.append(f"(Z[{parent.id}->{kc.id}])0")
                    self.bn0.addArc(f"({parent.id})0", f"(Z[{parent.id}->{kc.id}])0")
                    self.bn0.cpt(f"(Z[{parent.id}->{kc.id}])0")[{f"({parent.id})0": 0}] = [1 - s, s]
                    self.bn0.cpt(f"(Z[{parent.id}->{kc.id}])0")[{f"({parent.id})0": 1}] = [1 - c, c]
                    self.bn0.addArc(f"(Z[{parent.id}->{kc.id}])0", f"({kc.id})0")

                    self.tbn.add(
                        gum.LabelizedVariable(f"(Z[{parent.id}->{kc.id}])0", f"(Z[{parent.id}->{kc.id}])0", 2))
                    self.tbn_nodes.append(f"(Z[{parent.id}->{kc.id}])0")
                    self.tbn.addArc(f"({parent.id})0", f"(Z[{parent.id}->{kc.id}])0")
                    self.tbn.cpt(f"(Z[{parent.id}->{kc.id}])0")[{f"({parent.id})0": 0}] = [1 - s, s]
                    self.tbn.cpt(f"(Z[{parent.id}->{kc.id}])0")[{f"({parent.id})0": 1}] = [1 - c, c]
                    self.tbn.addArc(f"(Z[{parent.id}->{kc.id}])0", f"({kc.id})0")

                    # ... and a node (Z[X->Y])t
                    self.tbn.add(
                        gum.LabelizedVariable(f"(Z[{parent.id}->{kc.id}])t", f"(Z[{parent.id}->{kc.id}])t", 2))
                    self.tbn_nodes.append(f"(Z[{parent.id}->{kc.id}])t")
                    self.tbn.addArc(f"({parent.id})t", f"(Z[{parent.id}->{kc.id}])t")
                    self.tbn.cpt(f"(Z[{parent.id}->{kc.id}])t")[{f"({parent.id})t": 0}] = [1 - s, s]
                    self.tbn.cpt(f"(Z[{parent.id}->{kc.id}])t")[{f"({parent.id})t": 1}] = [1 - c, c]
                    self.tbn.addArc(f"(Z[{parent.id}->{kc.id}])t", f"({kc.id})t")

    def extend_evidence(self, evs):
        return evs

    def is_state_valid(self, state):
        nodes = list(state.keys())

        for node in [n for n in nodes if determine_node_type(n) == 'mastery']:
            kc_id = int(re.search('\((.+?)\)', node).group(1))
            timestamp = determine_node_timestamp(node)
            parent_nodes = [n for n in nodes if n.endswith(f"{kc_id}]){timestamp}")]

            if parent_nodes:
                if state[node] == 1 and any((1 - state[n] for n in parent_nodes)):
                    return False
                elif state[node] == 0 and all((state[n] for n in parent_nodes)):
                    return False

        return True

    def get_subevidence_from_block(self, block_name, full_state, evidence):
        n_eval = get_max_timestamp(evidence) - 1
        t = determine_node_timestamp(block_name)
        block_id = int(re.search('B\[(.+?)\]', block_name).group(1))

        if t == 0:
            outer_nodes = [f"({parent.id})0" for parent in self.learner_pool.get_learner_pool_kc_parents(block_id)] + [
                f"(T[{block_id}])1"
            ]
        elif t == n_eval:
            outer_nodes = [
                f"({parent.id}){t}" for parent in self.learner_pool.get_learner_pool_kc_parents(block_id)] + [
                f"({block_id}){t-1}"] + [
                f"(Z[{block_id}->{child.id}]){t}" for child in self.learner_pool.get_learner_pool_kc_children(block_id)
            ]
        else:
            outer_nodes = [
                f"({parent.id}){t}" for parent in self.learner_pool.get_learner_pool_kc_parents(block_id)] + [
                f"({block_id}){t-1}", f"(T[{block_id}]){t+1}"] + [
                f"(Z[{block_id}->{child.id}]){t}" for child in self.learner_pool.get_learner_pool_kc_children(block_id)
            ]

        subevidence = {**{n: evidence[n] for n in evidence.keys()},
                       **{n: full_state[n] for n in outer_nodes if n not in evidence.keys()}
                       }
        return subevidence

    def update_state_from_block_inference(self, previous_state, block_name, evidence):
        # on créé le block à partir de son nom
        n_eval = get_max_timestamp(evidence) - 1
        t = determine_node_timestamp(block_name)
        block_id = int(re.search('B\[(.+?)\]', block_name).group(1))

        if t == 0:  # si c'est t=0
            bn = self.get_initial_block_bn(block_id, evidence)
        elif t == n_eval:  # si c'est t=n_eval
            bn = self.get_final_block_bn(block_id, t, evidence)
        else:  # sinon
            bn = self.get_other_block_bn(block_id, t, evidence)

        nodes = bn.names()
        inner_nodes = {node for node in nodes if node not in evidence.keys()}
        block_evidence = {n: evidence[n] for n in nodes if n in evidence.keys()}
        if all((n in block_evidence for n in nodes)):
            state = previous_state
            del bn
        else:
            ie = gum.LazyPropagation(bn)
            ie.addJointTarget(nodes)
            ie.setEvidence(block_evidence)
            ie.makeInference()

            posterior = ie.jointPosterior(inner_nodes)
            values = posterior.toarray()
            var_names = posterior.var_names
            tt = truthtable(len(var_names))
            combinations = [{var_names[i]: int(elt[i]) for i in range(len(elt))} for elt in tt]
            block_pba_values = np.array([[comb, values[tuple(list(comb.values()))]] for comb in combinations])
            block_state = random.choices(block_pba_values[:, 0], weights=block_pba_values[:, 1])[0]

            state = {n: int(block_state[n]) if n in block_state.keys() else previous_state[n] for n in previous_state.keys()}
            del posterior, ie, bn

        return state

    def get_initial_block_bn(self, block_id, evidence):
        bn = gum.BayesNet()
        block_kc = self.learner_pool.get_kc_from_id(block_id)

        prior = self.learner_pool.get_prior(block_kc)
        learn = self.learner_pool.get_learn(block_kc)
        forget = self.learner_pool.get_forget(block_kc)

        # Introduce the structure of the temporal relationships between same KC's nodes
        # for all knowledge component X, there is a node (X)0...
        if self.learner_pool.has_kc_parents(block_kc):
            bn.addAND(gum.LabelizedVariable(f"({block_kc.id})0", f"({block_kc.id})0", 2))
        else:
            bn.add(gum.LabelizedVariable(f"({block_kc.id})0", f"({block_kc.id})0", 2))
            bn.cpt(f"({block_kc.id})0").fillWith([1 - prior, prior])

        # ... a node (T[X])t linked to (X)0...
        bn.add(gum.LabelizedVariable(f"(T[{block_kc.id}])1", f"(T[{block_kc.id}])1", 2))
        bn.addArc(f"({block_kc.id})0", f"(T[{block_kc.id}])1")
        bn.cpt(f"(T[{block_kc.id}])1")[{f"({block_kc.id})0": 0}] = [1 - learn, learn]
        bn.cpt(f"(T[{block_kc.id}])1")[{f"({block_kc.id})0": 1}] = [forget, 1 - forget]

        parents = self.learner_pool.get_learner_pool_kc_parents(block_kc)
        if parents:
            for parent in parents:
                c, s = self.learner_pool.get_c_param(parent, block_kc), self.learner_pool.get_s_param(parent, block_kc)
                bn.add(gum.LabelizedVariable(f"({parent.id})0", f"({parent.id})0", 2))
                bn.cpt(f"({parent.id})0").fillWith([1 - evidence[f"({parent.id})0"], evidence[f"({parent.id})0"]])
                bn.add(
                    gum.LabelizedVariable(f"(Z[{parent.id}->{block_kc.id}])0", f"(Z[{parent.id}->{block_kc.id}])0", 2))
                bn.addArc(f"({parent.id})0", f"(Z[{parent.id}->{block_kc.id}])0")
                bn.cpt(f"(Z[{parent.id}->{block_kc.id}])0")[{f"({parent.id})0": 0}] = [1 - s, s]
                bn.cpt(f"(Z[{parent.id}->{block_kc.id}])0")[{f"({parent.id})0": 1}] = [1 - c, c]
                bn.addArc(f"(Z[{parent.id}->{block_kc.id}])0", f"({block_kc.id})0")

        children = self.learner_pool.get_learner_pool_kc_children(block_kc)
        if children:
            for child in children:
                c, s = self.learner_pool.get_c_param(block_kc, child), self.learner_pool.get_s_param(block_kc, child)
                bn.add(
                    gum.LabelizedVariable(f"(Z[{block_kc.id}->{child.id}])0", f"(Z[{block_kc.id}->{child.id}])0", 2))
                bn.addArc(f"({block_kc.id})0", f"(Z[{block_kc.id}->{child.id}])0")
                bn.cpt(f"(Z[{block_kc.id}->{child.id}])0")[{f"({block_kc.id})0": 0}] = [1 - s, s]
                bn.cpt(f"(Z[{block_kc.id}->{child.id}])0")[{f"({block_kc.id})0": 1}] = [1 - c, c]

        bn.add(gum.LabelizedVariable(f"(O[{block_kc.id}])0", f"(O[{block_kc.id}])0", 2))
        bn.addArc(f"({block_kc.id})0", f"(O[{block_kc.id}])0")
        guess, slip = self.learner_pool.get_guess(block_kc.id), self.learner_pool.get_slip(block_kc.id)
        bn.cpt(f"(O[{block_kc.id}])0")[{f"({block_kc.id})0": 0}] = [1 - guess, guess]
        bn.cpt(f"(O[{block_kc.id}])0")[{f"({block_kc.id})0": 1}] = [slip, 1 - slip]

        return bn

    def get_final_block_bn(self, block_id, n_final, evidence):
        bn = gum.BayesNet()
        block_kc = self.learner_pool.get_kc_from_id(block_id)

        learn = self.learner_pool.get_learn(block_kc)
        forget = self.learner_pool.get_forget(block_kc)

        # Introduce the structure of the temporal relationships between same KC's nodes
        # for all knowledge component X, there is a node (X)0...
        bn.addAND(gum.LabelizedVariable(f"({block_kc.id}){n_final}", f"({block_kc.id}){n_final}", 2))

        # ... a node (T[X])t linked to (X)0...
        bn.add(gum.LabelizedVariable(f"({block_kc.id}){n_final-1}", f"({block_kc.id}){n_final-1}", 2))
        bn.cpt(f"({block_kc.id}){n_final - 1}").fillWith([1 - evidence[f"({block_kc.id}){n_final - 1}"],
                                                            evidence[f"({block_kc.id}){n_final - 1}"]])

        bn.add(gum.LabelizedVariable(f"(T[{block_kc.id}]){n_final}", f"(T[{block_kc.id}]){n_final}", 2))
        bn.addArc(f"(T[{block_kc.id}]){n_final}", f"({block_kc.id}){n_final}")
        bn.addArc(f"({block_kc.id}){n_final-1}", f"(T[{block_kc.id}]){n_final}")
        bn.cpt(f"(T[{block_kc.id}]){n_final}")[{f"({block_kc.id}){n_final-1}": 0}] = [1 - learn, learn]
        bn.cpt(f"(T[{block_kc.id}]){n_final}")[{f"({block_kc.id}){n_final-1}": 1}] = [forget, 1 - forget]


        parents = self.learner_pool.get_learner_pool_kc_parents(block_kc)
        if parents:
            for parent in parents:
                c, s = self.learner_pool.get_c_param(parent, block_kc), self.learner_pool.get_s_param(parent, block_kc)
                bn.add(gum.LabelizedVariable(f"({parent.id}){n_final}", f"({parent.id}){n_final}", 2))
                bn.cpt(f"({parent.id}){n_final}").fillWith([1 - evidence[f"({parent.id}){n_final}"],
                                                            evidence[f"({parent.id}){n_final}"]])
                bn.add(
                    gum.LabelizedVariable(
                        f"(Z[{parent.id}->{block_kc.id}]){n_final}", f"(Z[{parent.id}->{block_kc.id}]){n_final}", 2))
                bn.addArc(f"({parent.id}){n_final}", f"(Z[{parent.id}->{block_kc.id}]){n_final}")
                bn.cpt(f"(Z[{parent.id}->{block_kc.id}]){n_final}")[{f"({parent.id}){n_final}": 0}] = [1 - s, s]
                bn.cpt(f"(Z[{parent.id}->{block_kc.id}]){n_final}")[{f"({parent.id}){n_final}": 1}] = [1 - c, c]
                bn.addArc(f"(Z[{parent.id}->{block_kc.id}]){n_final}", f"({block_kc.id}){n_final}")

        children = self.learner_pool.get_learner_pool_kc_children(block_kc)
        if children:
            for child in children:
                c, s = self.learner_pool.get_c_param(block_kc, child), self.learner_pool.get_s_param(block_kc, child)
                bn.add(
                    gum.LabelizedVariable(
                        f"(Z[{block_kc.id}->{child.id}]){n_final}", f"(Z[{block_kc.id}->{child.id}]){n_final}", 2))
                bn.addArc(f"({block_kc.id}){n_final}", f"(Z[{block_kc.id}->{child.id}]){n_final}")
                bn.cpt(f"(Z[{block_kc.id}->{child.id}]){n_final}")[{f"({block_kc.id}){n_final}": 0}] = [1 - s, s]
                bn.cpt(f"(Z[{block_kc.id}->{child.id}]){n_final}")[{f"({block_kc.id}){n_final}": 1}] = [1 - c, c]

        bn.add(gum.LabelizedVariable(f"(O[{block_kc.id}]){n_final}", f"(O[{block_kc.id}]){n_final}", 2))
        bn.addArc(f"({block_kc.id}){n_final}", f"(O[{block_kc.id}]){n_final}")
        guess, slip = self.learner_pool.get_guess(block_kc.id), self.learner_pool.get_slip(block_kc.id)
        bn.cpt(f"(O[{block_kc.id}]){n_final}")[{f"({block_kc.id}){n_final}": 0}] = [1 - guess, guess]
        bn.cpt(f"(O[{block_kc.id}]){n_final}")[{f"({block_kc.id}){n_final}": 1}] = [slip, 1 - slip]

        return bn

    def get_other_block_bn(self, block_id, timestamp, evidence):
        bn = gum.BayesNet()
        block_kc = self.learner_pool.get_kc_from_id(block_id)

        learn = self.learner_pool.get_learn(block_kc)
        forget = self.learner_pool.get_forget(block_kc)

        # Introduce the structure of the temporal relationships between same KC's nodes
        # for all knowledge component X, there is a node (X)0...
        bn.addAND(gum.LabelizedVariable(f"({block_kc.id}){timestamp}", f"({block_kc.id}){timestamp}", 2))

        # ... a node (T[X])t linked to (X)0...
        bn.add(gum.LabelizedVariable(f"({block_kc.id}){timestamp - 1}", f"({block_kc.id}){timestamp - 1}", 2))
        bn.cpt(f"({block_kc.id}){timestamp - 1}").fillWith([1 - evidence[f"({block_kc.id}){timestamp - 1}"],
                                                            evidence[f"({block_kc.id}){timestamp - 1}"]])

        bn.add(gum.LabelizedVariable(f"(T[{block_kc.id}]){timestamp}", f"(T[{block_kc.id}]){timestamp}", 2))
        bn.add(gum.LabelizedVariable(f"(T[{block_kc.id}]){timestamp+1}", f"(T[{block_kc.id}]){timestamp+1}", 2))

        bn.addArc(f"({block_kc.id}){timestamp-1}", f"(T[{block_kc.id}]){timestamp}")
        bn.addArc(f"(T[{block_kc.id}]){timestamp}", f"({block_kc.id}){timestamp}")
        bn.addArc(f"({block_kc.id}){timestamp}", f"(T[{block_kc.id}]){timestamp + 1}")

        bn.cpt(f"(T[{block_kc.id}]){timestamp+1}")[{f"({block_kc.id}){timestamp}": 0}] = [1 - learn, learn]
        bn.cpt(f"(T[{block_kc.id}]){timestamp+1}")[{f"({block_kc.id}){timestamp}": 1}] = [forget, 1 - forget]

        bn.cpt(f"(T[{block_kc.id}]){timestamp}")[{f"({block_kc.id}){timestamp - 1}": 0}] = [1 - learn, learn]
        bn.cpt(f"(T[{block_kc.id}]){timestamp}")[{f"({block_kc.id}){timestamp - 1}": 1}] = [forget, 1 - forget]

        parents = self.learner_pool.get_learner_pool_kc_parents(block_kc)
        if parents:
            for parent in parents:
                c, s = self.learner_pool.get_c_param(parent, block_kc), self.learner_pool.get_s_param(parent, block_kc)
                bn.add(gum.LabelizedVariable(f"({parent.id}){timestamp}", f"({parent.id}){timestamp}", 2))
                bn.cpt(f"({parent.id}){timestamp}").fillWith([1 - evidence[f"({parent.id}){timestamp}"],
                                                              evidence[f"({parent.id}){timestamp}"]])
                bn.add(
                    gum.LabelizedVariable(
                        f"(Z[{parent.id}->{block_kc.id}]){timestamp}", f"(Z[{parent.id}->{block_kc.id}]){timestamp}", 2))
                bn.addArc(f"({parent.id}){timestamp}", f"(Z[{parent.id}->{block_kc.id}]){timestamp}")
                bn.cpt(f"(Z[{parent.id}->{block_kc.id}]){timestamp}")[{f"({parent.id}){timestamp}": 0}] = [1 - s, s]
                bn.cpt(f"(Z[{parent.id}->{block_kc.id}]){timestamp}")[{f"({parent.id}){timestamp}": 1}] = [1 - c, c]
                bn.addArc(f"(Z[{parent.id}->{block_kc.id}]){timestamp}", f"({block_kc.id}){timestamp}")

        children = self.learner_pool.get_learner_pool_kc_children(block_kc)
        if children:
            for child in children:
                c, s = self.learner_pool.get_c_param(block_kc, child), self.learner_pool.get_s_param(block_kc, child)
                bn.add(
                    gum.LabelizedVariable(
                        f"(Z[{block_kc.id}->{child.id}]){timestamp}", f"(Z[{block_kc.id}->{child.id}]){timestamp}", 2))
                bn.addArc(f"({block_kc.id}){timestamp}", f"(Z[{block_kc.id}->{child.id}]){timestamp}")
                bn.cpt(f"(Z[{block_kc.id}->{child.id}]){timestamp}")[{f"({block_kc.id}){timestamp}": 0}] = [1 - s, s]
                bn.cpt(f"(Z[{block_kc.id}->{child.id}]){timestamp}")[{f"({block_kc.id}){timestamp}": 1}] = [1 - c, c]

        bn.add(gum.LabelizedVariable(f"(O[{block_kc.id}]){timestamp}", f"(O[{block_kc.id}]){timestamp}", 2))
        bn.addArc(f"({block_kc.id}){timestamp}", f"(O[{block_kc.id}]){timestamp}")
        guess, slip = self.learner_pool.get_guess(block_kc.id), self.learner_pool.get_slip(block_kc.id)
        bn.cpt(f"(O[{block_kc.id}]){timestamp}")[{f"({block_kc.id}){timestamp}": 0}] = [1 - guess, guess]
        bn.cpt(f"(O[{block_kc.id}]){timestamp}")[{f"({block_kc.id}){timestamp}": 1}] = [slip, 1 - slip]

        return bn
