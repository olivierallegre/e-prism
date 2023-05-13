import sys
import time
sys.path.append('..')
import numpy as np
import pandas as pd
from src.knowledge_components import KnowledgeComponent, SetOfKnowledgeComponents, \
    get_set_of_knowledge_components_from_dataset
from src.exercises import Exercise
from src.prerequisite_links import PrerequisiteLink, SetOfPrerequisiteLinks
from src.learners import Learner, LearnerPool
from pyBKT.models import Model
import argparse
import copy
from src.utils import get_strongest_folds
import tqdm
import multiprocessing
import json
from pathlib import Path


def kc_parameters_to_str(parameters):
    str_params = {}
    for param_key in ('prior', 'learn', 'forget'):
        str_params[param_key] = {key.name: parameters[param_key][key] for key in parameters[param_key].keys()}

    for param_key in ('c', 's'):
        str_params[param_key] = {}
        for first_key in parameters[param_key].keys():
            for second_key in parameters[param_key][first_key]:
                str_params[param_key][f"{second_key.name}->{first_key.name}"] = parameters[param_key][first_key][second_key]
    return str_params


def run(set_of_knowledge_components, set_of_prerequisite_links, df_train, inference_model, fit_model, n_gibbs,
                   burn_in, verbose, n_iter):
    #print(f"{n_iter} began")

    folds = get_strongest_folds(df_train)
    aucs, rmses = [], []
    best_params = None
    best_rmse = 1

    for i, fold in tqdm.tqdm(enumerate(folds), total=5):
        learner_pool = LearnerPool(set_of_knowledge_components,
                                   set_of_prerequisite_links,
                                   inference_model_type=inference_model)

        test_ids = fold
        train_ids = list(set(list(df_train.index.values)) - set(test_ids))
        test_df = df_train.iloc[test_ids]
        train_df = df_train.iloc[train_ids]

        learner_pool.setup_random_parameters()
        if verbose == 2:
            print("\n initial parameters")
            learner_pool.print_parameters()

        learner_pool.fit(train_df, defaults="as",fit_model=fit_model, n_gibbs=n_gibbs, burn_in_period=burn_in,  verbose=verbose)
        metrics = learner_pool.evaluate(data=test_df, defaults="as", metrics=['auc', 'rmse'])
        auc_test, rmse_test = metrics['auc'], metrics['rmse']
        params = learner_pool.params
        del learner_pool

        if rmse_test < best_rmse:
            best_params = params
            best_rmse = rmse_test

        aucs.append(auc_test)
        rmses.append(rmse_test)

    return {'train_aucs': aucs, 'train_rmses': rmses, 'best_rmse': best_rmse, 'best_params': best_params}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Evaluate the E-PRISM learner model from data")

    parser.add_argument("--dataset",
                        type=str,
                        default="assistments_58_74_277",
                        help="the path to the data")

    parser.add_argument("--source",
                        type=str,
                        default="",
                        help="name of the source KC of studied prerequisite link")

    parser.add_argument("--target",
                        type=str,
                        default="",
                        help="name of the target KC of the studied prerequisite link")

    parser.add_argument("--fit_model",
                        type=str,
                        default="mcem",
                        help="the parameter learning algorithm")

    parser.add_argument("--inference_model",
                        type=str,
                        default="NoisyAND",
                        help="the kind of inference model")

    parser.add_argument("--min_iter",
                        type=int,
                        default=2,
                        help="min number of iterations")

    parser.add_argument("--max_iter",
                        type=int,
                        default=3,
                        help="max number of iterations")

    parser.add_argument("--n_gibbs",
                        type=int,
                        default=10,
                        help="number of init")

    parser.add_argument("--burn_in",
                        type=int,
                        default=0,
                        help="burn-in period")

    parser.add_argument("--n_init",
                        type=int,
                        default=10,
                        help="number of EM initializations")

    parser.add_argument("--metric",
                        type=str,
                        default="rmse",
                        help="metric to print")

    parser.add_argument("-v",
                        type=int,
                        default=1,
                        help="verbosity mode [0, 1, 2].")

    parser.add_argument("--defaults",
                        type=str,
                        default="as",
                        help="corresponding dict.")

    parser.add_argument("--log_dir",
                        type=str,
                        default="logs",
                        help="log dir.")

    args = parser.parse_args()
    Path(f'~/data/{args.dataset}/{args.log_dir}').mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(f"~/data/{args.dataset}/processed_data_train_{args.max_iter}interactions.csv", sep='\t')
    test_df = pd.read_csv(f"~/data/{args.dataset}/processed_data_test_{args.max_iter}interactions.csv", sep='\t')

    # DATA TO CONTINUOUS LEARNER TRACES
    set_of_knowledge_components = get_set_of_knowledge_components_from_dataset(
        pd.concat([train_df, test_df], ignore_index=True, sort=False), defaults=args.defaults
    )

    if args.source and args.target:
        source = set_of_knowledge_components.get_kc_from_id(int(args.source))
        target = set_of_knowledge_components.get_kc_from_id(int(args.target))
        set_of_prerequisite_links = SetOfPrerequisiteLinks([PrerequisiteLink(source=source, target=target)])
    else:
        set_of_prerequisite_links = SetOfPrerequisiteLinks([])


    result_list = []

    def log_result(result):
        # This is called whenever foo_pool(i) returns a result.
        # result_list is modified only by the main process, not the pool workers.
        result_list.append(result)

    pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
    res = []
    for i in range(args.n_init):
        res.append(
            pool.apply_async(run,
                         args=(set_of_knowledge_components, set_of_prerequisite_links, train_df, args.inference_model,
                               args.fit_model, args.n_gibbs, args.burn_in, args.v, i),
                         callback=log_result)
        )

    pool.close()
    for r in res:
        r.get()

    pool.join()

    best_rmse = min([res_elt["best_rmse"] for res_elt in result_list])
    best_parameters = [res_elt['best_params'] for res_elt in result_list if res_elt['best_rmse'] == best_rmse][0]

    if args.v > 0:
        print(kc_parameters_to_str(best_parameters))

    test_learner_pool = LearnerPool(set_of_knowledge_components, set_of_prerequisite_links)
    test_learner_pool.set_parameters(best_parameters)
    metric = test_learner_pool.evaluate(test_df, metrics=['auc', 'rmse'])

    print("Results E-PRISM prediction")
    print("mean training rmse", np.mean([res_elt["train_rmses"] for res_elt in result_list]))
    print("best training RMSE", max([res_elt["best_rmse"] for res_elt in result_list]))
    print("mean training AUC", np.mean([res_elt["train_aucs"] for res_elt in result_list]))
    print("best training AUC", max(max([res_elt["train_aucs"] for res_elt in result_list])))

    print('Eval RMSE = ', metric['rmse'])
    print("Eval AUC = ", metric['auc'])
