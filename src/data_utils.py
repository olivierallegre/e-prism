import pandas as pd
import numpy as np
from knowledge_components import KnowledgeComponent, SetOfKnowledgeComponents
from prerequisite_links import PrerequisiteLink, SetOfPrerequisiteLinks
from exercises import Exercise
from learners import Learner, LearnerTrace, LearnerPool


def load_dataset(dataset):
    assert any((isinstance(dataset, str), isinstance(dataset, pd.DataFrame))), "dataset must be str or dataframe"
    df = dataset if isinstance(dataset, pd.DataFrame) else pd.read_csv(dataset)
    return df


def get_knowledge_components_and_exercises_from_dataset(traces_df, defaults):
    """
    Return the set of knowledge components and the set of exercises from a given dataset path or df
    :param traces_df: pd.DataFrame, the Dataframe of the traces dataset
    :param defaults: dict, transcription of the dataset columns
    :return: tuple, composed of the set of knowledge components and the set of exercises
    """
    # For each row, we extract the id of the exercise done and the KC associated to it
    knowledge_components, exercises = [], []
    for i, row in traces_df.iterrows():
        kc_id = row[defaults['kc_id']]
        if kc_id not in [kc.id for kc in knowledge_components]:
            kc_name = row[defaults['kc_name']] if defaults['kc_name'] in traces_df.columns else kc_id
            kc = KnowledgeComponent(kc_id, kc_name)
            knowledge_components.append(kc)
        else:
            kc = [kc for kc in knowledge_components if kc.id == kc_id][0]
        exercise_id = row[defaults['exercise_id']]
        if exercise_id not in [exercise.id for exercise in exercises]:
            exercise = Exercise(exercise_id, kc)
            exercises.append(exercise)
    return SetOfKnowledgeComponents(knowledge_components), exercises


def get_prerequisite_links_from_csv(prerequisites_csv_file, set_of_knowledge_components):
    """
    Get the set of prerequisite links described in a given csv file.
    :param prerequisites_csv_file: csv file that describes the prerequisite links
    :param set_of_knowledge_components: the set of KCs where prerequisite are from
    :return: SetOfPrerequisiteLinks, the set of prerequisite links declared in the dataset
    """
    df = load_dataset(prerequisites_csv_file)
    assert all(('source_id', 'target_id', 'strength')) in df.keys(), "Missing entries in prerequisite links dataset."
    prerequisite_links = []
    for i, row in df.iterrows():
        source_kc = set_of_knowledge_components.get_kc_from_id(row['source_id'])
        target_kc = set_of_knowledge_components.get_kc_from_id(row['target_id'])
        strength = row['strength']
        prerequisite_links.append(PrerequisiteLink(source_kc, target_kc, strength))
    return SetOfPrerequisiteLinks(prerequisite_links)


def get_learners_and_learner_traces_from_traces_and_prerequisites_dataset(traces_dataset, prerequisites_csv,
                                                                          defaults=None):
    """
    Return the set of knowledge components and the set of exercises from a given dataset path or df
    :param traces_dataset: str or pd.DataFrame, the dataset of the learner traces
    :param prerequisites_csv: str or pd.DataFrame, the dataset of the learner traces
    :param defaults: dict, transcription of the dataset columns
    :return: dict
    """
    df = load_dataset(traces_dataset)

    if defaults is None:
        defaults = {'order_id': 'index',
                    'kc_id': 'kc_id',
                    'kc_name': 'kc_id',
                    'correct': 'success',
                    'user_id': 'learner_id',
                    'exercise_id': 'exercise_id'
                    }

    knowledge_components, exercises = get_knowledge_components_and_exercises_from_dataset(df, defaults)
    prerequisite_links = get_prerequisite_links_from_csv(prerequisites_csv, knowledge_components)

    # For now, we suppose the learner pool is the same for every learner in the learner traces dataset.
    # We must keep in mind that it might not be the case.
    learner_pool = LearnerPool(knowledge_components, prerequisite_links)
    learners = [Learner(learner_id, learner_pool) for learner_id in np.unique(df[defaults['user_id']])]
    learner_traces = {}
    for learner in learners:
        learner_df = df[df[defaults['user_id']] == learner.id]
        learner_traces[learner] = []
        for i, row in learner_df.iterrows():
            exercise = next((x for x in exercises if x.id == row[defaults['exercise_id']]), None)
            if exercise is None:
                return Exception('exercise is none')
            success = bool(row[defaults['correct']])
            trace = LearnerTrace(learner, exercise, success)
            learner_traces[learner].append(trace)
    return learner_traces
