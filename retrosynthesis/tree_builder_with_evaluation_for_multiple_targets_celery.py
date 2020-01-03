
from makeit.retrosynthetic.mcts.tree_builder import MCTS
from makeit.synthetic.evaluation.tree_evaluator import TreeEvaluator
from makeit.utilities.io.logger import MyLogger
import makeit.global_config as gc
import pandas as pd
from makeit.utilities.io import name_parser
from rdkit import Chem
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import pprint


MyLogger.initialize_logFile()

def get_chemical_table_from_routes(routes, contexts):
    # all_chemicals = get_all_chemicals_in_all_routes(routes)
    # print(all_chemicals)
    # print(len(all_chemicals))
    all_routes_chemical_only = []
    all_routes_target = []
    all_routes_score = []
    for route in routes:
        route_target = route['chemicals'][0]['SMILES']
        route_chemical_only = [chemical['SMILES'] for chemical in route['chemicals'] if chemical['is_starting']]
        all_routes_chemical_only.append(route_chemical_only)
        all_routes_target.append(route_target)
        all_routes_score.append(route['score'])
    ohe = MultiLabelBinarizer()
    ohe_context = MultiLabelBinarizer()
    # ohe.fit(all_chemicals)
    # print('transformed')
    all_routes_chemical_onehot = ohe.fit_transform(all_routes_chemical_only)
    all_routes_context_onehot = ohe_context.fit_transform(contexts)
    chemical_smiles = list(ohe.classes_)
    context_smiles = list(ohe_context.classes_)
    return (all_routes_target, all_routes_chemical_onehot, chemical_smiles,all_routes_score, all_routes_context_onehot, context_smiles)

def get_chemicals_from_tree(tree, path = [],):
    path.append({'SMILES':tree['smiles'],
                 'is_starting': False})
    if not tree['children']:
        path[-1]['is_starting']=True
        pass
    else:
        for child in tree['children']:
            [get_chemicals_from_tree(child_chem, path) for child_chem in child['children']]
    return path

def get_context_from_tree(tree, path = [],):
    if not tree['children']:
        pass
    else:
        for child in tree['children']:
            context = child['context']
            [path.append(context[i]) for i in [1,2,3] if context[i]!='']
            [get_context_from_tree(child_chem, path) for child_chem in child['children']]
    return list(set(path))


print('start to load targets')
# dataset = pd.read_csv('common_substructure.csv')
# dataset = pd.read_csv('Top30DrugsByPrescription.csv')
# dataset.head()

# with open('truncated_mol_lib.pickle','r') as MOLLIB:
with open('WHO_truncated.pickle','r') as MOLLIB:
    truncated_mol_lib = pickle.load(MOLLIB)


print(len(truncated_mol_lib))# for name in dataset['SMILES']:
#     try:
#         mol = name_parser.name_to_molecule(name)
#         mol_lib.append(mol)
#     except:
#         continue
print("{} target molecules to expand".format(len(truncated_mol_lib)))
celery = False
NCPUS = 4
# treeBuilder = TreeBuilder(celery=celery, mincount=25, mincount_chiral=10)
Tree = MCTS(nproc=NCPUS, mincount=gc.RETRO_TRANSFORMS_CHIRAL['mincount'], 
        mincount_chiral=gc.RETRO_TRANSFORMS_CHIRAL['mincount_chiral'],
        celery=celery)
treeEvaluator = TreeEvaluator(context_recommender=gc.neural_network, celery=celery)
all_routes = []
all_trees = []
all_routes_contexts = []
index_list_all_routes = []
is_first_target = True
status = 0
results_folder = "/home/hanyug/Combined_synthesis_planning/pathways/small_and_buyable/"

status_file = open(results_folder+'status for molecules.dat','w')
status_file.write('SMILES\tnumberofpaths\tnotes\n')
reaction_dict = {}
for mol in truncated_mol_lib:
# for mol in [Chem.MolFromSmiles('OC(C1CCCCN1)c2cc(nc3c2cccc3C(F)(F)F)C(F)(F)F')]:
    if is_first_target:
        soft_reset = False
        is_first_target = False
    else:
        soft_reset = True
# for mol in [Chem.MolFromSmiles('O=C1CN=C(c2ccccn2)c2cc(Br)ccc2N1')]:
    try:
        smiles = Chem.MolToSmiles(mol)   
        print('expanding target {}'.format(smiles)) 
        #require starting materials small and cheap
        # status, paths = treeBuilder.get_buyable_paths(smiles, max_depth=10, template_prioritization=gc.relevance,
        #                                     precursor_prioritization=gc.scscore, nproc=1, expansion_time=300, max_trees=1000, 
        #                                     max_branching=25, apply_fast_filter=True, filter_threshold=0.9,
        #                                     max_ppg=1000,
        #                                     max_natom_dict={'C':10,'N':3,'O':3,'logic': 'and'},
        #                                     min_chemical_history_dict={'as_reactant':1, 'as_product':1, 'logic':'and'})
        status, paths = Tree.get_buyable_paths(smiles,
                                            nproc=NCPUS,
                                            max_depth=10,
                                            expansion_time=100,
                                            max_trees=10000, 
                                            max_cum_template_prob=0.995,
                                            max_branching=25, apply_fast_filter=True, filter_threshold=0.75,
                                            template_count=1000,
                                            max_ppg = 100, 
                                            max_natom_dict={'C':10,'N':3,'O':5,'logic': 'and'},
                                            # min_chemical_history_dict={'as_reactant':1, 'as_product':1,'logic':'and'},
                                            soft_reset=soft_reset,
                                            soft_stop=True)
        print('done for target {}'.format(smiles))

        for chemical in Tree.Chemicals:
            chemical_obj = Tree.Chemicals[chemical]
            for _id, result in chemical_obj.template_idx_results.items():
                if result.waiting:
                    continue
                for rsmi in result.reactions:
                    rxn = rsmi+'>>'+chemical
                    if rxn not in reaction_dict:
                        reaction_dict[rxn]={}
                        for rct in rsmi.split('.'):
                            reaction_dict[rxn][rct]=-1
                        reaction_dict[rxn][chemical]=1
        # trees = [{'is_chemical': True, 'smiles': 'CN1C2CCC1CC(C2)OC(=O)C(CO)c3ccccc3', 'ppg': 0.0, 'id': 1, 'children': [{'info': '', 'smiles': 'CN1C2CCC1CC(O)C2.O=C(O)C(CO)c1ccccc1>>CN1C2CCC1CC(C2)OC(=O)C(CO)c3ccccc3', 'is_reaction': True, 'num_examples': 19578, 'template_score': 0.017628178000450134, 'children': [
        #     {'is_chemical': True, 'smiles': 'CN1C2CCC1CC(O)C2', 'ppg': 1.0, 'id': 3, 'children': []}, {'is_chemical': True, 'smiles': 'O=C(O)C(CO)c1ccccc1', 'ppg': 1.0, 'id': 4, 'children': []}], 'id': 2, 'necessary_reagent': u''}]}]
        print('number of reactions explored: {}'.format(len(reaction_dict)))
        feasible_trees = []
        for tree in paths:
            res = treeEvaluator.evaluate_tree(tree, gc.neural_network, gc.probability,
                               gc.templatefree, gc.forwardonly, is_target=True, reset=False, nproc=2, n=5)
            feasible_trees.append(res)
        routes_for_one_target = [{'chemicals':get_chemicals_from_tree(tree['tree'],path = []),'score':tree['score']} for tree in feasible_trees if tree['tree']['children']]
        context_for_one_target = [get_context_from_tree(tree['tree'],path = []) for tree in feasible_trees if tree['tree']['children']]
        # print(context_for_one_target)
        #index the route
        # index_list_one_target = range(len(paths))
        # print feasible_trees
        all_routes.extend(routes_for_one_target)
        all_routes_contexts.extend(context_for_one_target)
        all_trees.extend([tree for tree in feasible_trees if tree['tree']['children']])
        status_file.write('{}\t{}\t{}\t\n'.format(smiles,len(feasible_trees),''))
    except Exception as e:
        status_file.write('{}\t{}\t{}\t\n'.format(smiles,0,e))
        pass    # index_list_all_routes.extend(index_list_one_target)
    print(len(all_routes))

# with open('entire_tree.pkl','w') as ET:
#     pickle.dump(Tree, ET)
with open(results_folder+'reaction_dict.pickle','w') as RD:
    pickle.dump(reaction_dict, RD)

status_file.close()

with open(results_folder+'evaluation_dict.pickle', 'w') as EVAL_DICT:
    pickle.dump(treeEvaluator.evaluation_dict, EVAL_DICT)
index_list_all_routes = range(len(all_routes))
with open(results_folder+'all_trees.pickle','w') as ROUTES_FILE:
    pickle.dump(all_trees, ROUTES_FILE)


