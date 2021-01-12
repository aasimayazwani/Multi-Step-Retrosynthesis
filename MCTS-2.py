from collections import defaultdict

class node:
    def __init__(self,parent,child,total,successes,mu_value):
        self.parent = parent 
        self.total_states = total
        self.children = child
        self.mu = mu_value
        self.successes = successes

Smiles4Hydroxybutyryl_CoA = 'CC(C)(COP(=O)(O)OP(=O)(O)OCC1C(C(C(O1)N2C=NC3=C(N=CN=C32)N)O)OP(=O)(O)O)C(C(=O)NCCC(=O)NCCSC(=O)CCCO)O'
Smiles4Hydroxybutyrate = 'C(CC(=O)O)CO'
SmilesSuccinylSemialdehyde = 'C(CC(=O)O)C=O'
SmilesSuccinyl_CoA = 'CC(C)(COP(=O)(O)OP(=O)(O)OCC1C(C(C(O1)N2C=NC3=C(N=CN=C32)N)O)OP(=O)(O)O)C(C(=O)NCCC(=O)NCCSC(=O)CCC(=O)O)O'
SmilesAlpha_KetoglutaricAcid = 'C(CC(=O)O)C(=O)C(=O)O'
SmilesSuccinicAcid = 'C(CC(=O)O)C(=O)O'
SmilesBDO14 = 'OCCCCO'


def simulation__children_trees(parent,parent_trees):
    #parent_trees[parent] = node(parent,0,0)
    child = parent_trees[parent].children
    for candidate in child:
        parent_trees[candidate] = node(candidate,0,0,0,0)    
    return parent_trees


class MCTS:
    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight
    
    def rollout_policy(react):
        # Heavy rollout vs light rollout 
        # choosing children based on machine learning values
        results =[]
        for i in range(0,len(react)):
            results.append(molecule_model_predict(react[i])[0][1])
        rollout_reward = reward_dict(react,results)
        temp = []
        for item in rollout_reward.keys():
            temp.append([item,rollout_reward[item]])
        s = np.array(temp)
        s = s[np.argsort(s[:, 1])]
        return s[-1][0]
    
    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node not in self.children:
            print(rollout_policy(node))
            return rollout_policy(node)
        else:
            return None
        

def rollout_policy(react):
    # Heavy rollout vs light rollout 
    # choosing children based on machine learning values
    results =[]
    for i in range(0,len(react)):
        results.append(molecule_model_predict(react[i])[0][1])
    rollout_reward = reward_dict(react,results)
    temp = []
    for item in rollout_reward.keys():
        temp.append([item,rollout_reward[item]])
    s = np.array(temp)
    s = s[np.argsort(s[:, 1])]
    return s[-1][0]

def tanimoto(initial,all_products):
    similarity_inital = []
    from rdkit.Chem import MACCSkeys
    for i in range(0,len(all_products)):
        mols = Chem.MolFromSmiles(initial),Chem.MolFromSmiles(all_products[i])
        fps = [ MACCSkeys.GenMACCSKeys(x) for x in mols ]
        similarity_inital.append([Chem.MolToSmiles(mols[1]),round(DataStructs.TanimotoSimilarity(fps[0], fps[1]), 4)])
    return similarity_inital

def tanimoto_filtered_children(children,parent,threshold=0.1):
    df_similarity_inital = pd.DataFrame(tanimoto(parent,children),columns = ["Products","Similarity"])
    tanimoto_filtering = df_similarity_inital[df_similarity_inital["Similarity"]>threshold]
    de_duplicated_smiles = list(tanimoto_filtering["Products"])
    return de_duplicated_smiles

def generate_children(parent,observed = [],hydrogen_add=True):
    return np.unique(state_space_generation(parent,rule_size,hydrogen_add)["products"])


def traverse(depth,react):
    observed = []
    while depth :
        a = rollout_policy(react)
        if a not in observed:
            observed.append(a)
        else:
            print(observed[-1])
            print("\n")
            #print("something wrong in code")
        depth -=1 
        react = generate_children(observed[0]) 
        react = list(np.setdiff1d(react,observed))
        react = tanimoto_filtered_children(parent=observed[0],children=react)
    return observed



def traverse(node):
    while fully_expanded(node): 
        node = best_uct(node) 
    return pick_univisted(node.children) or node  

def best_child(children):
    # Best child is the one with highest number of visits 
    # function  for picking the best child
    return None 

def backpropagate(node, result): 
    if is_root(node):
        return
    node.stats = update_stats(node, result)  
    backpropagate(node.parent)

def reward_dict(molecule,reward):
    temp = dict()
    for i in range(0,len(molecule)):
            temp[molecule[i]] = reward[i]
    return temp

def reward_dict(molecule,reward):
    temp = dict()
    for i in range(0,len(molecule)):
            temp[molecule[i]] = reward[i]
    return temp 

def rollout_policy(react):
    # Heavy rollout vs light rollout 
    # choosing children based on machine learning values
    results =[]
    for i in range(0,len(react)):
        results.append(molecule_model_predict(react[i])[0][1])
    rollout_reward = reward_dict(react,results)
    temp = []
    for item in rollout_reward.keys():
        temp.append([item,rollout_reward[item]])
    s = np.array(temp)
    s = s[np.argsort(s[:, 1])]
    return s[-1][0]

depth = 5 
def traverse(depth,react):
    "This would select the best candidate for each iteration."
    observed = []
    while depth :
        a = rollout_policy(react)
        if a not in observed:
            observed.append(a)
        else:
            print(observed[-1])
            print("\n")
            #print("something wrong in code")
        depth -=1 
        react = generate_children(observed[0]) 
        react = list(np.setdiff1d(react,observed))
        react = tanimoto_filtered_children(parent=observed[0],children=react)
    return observed

class node:
    def __init__(self,parent,child,total,successes,mu_value):
        self.parent = parent 
        self.total_states = total
        self.children = child
        self.mu = mu_value
        self.successes = successes
        
def generate_children(parent,observed = []):
    """
    Generates possible states. Given only information about the parent state and 
    knowledge of chidren nodes
    Input type: Smile string of parent node, list of visited states
    Return type: Numerical similarity score
    """
    return np.unique(state_space_generation(parent,rule_size,hydrogen_add)["products"])

def simulation_mu_value(parent,children_values):
    """Heavy rollout vs light rollout 
    choosing children based on machine learning values """
    react = tanimoto_filtered_children(parent=parent,children=children_values)
    results =[]
    for i in range(0,len(children_values)):
        results.append(molecule_model_predict(children_values[i],parent)[0][1])
    return np.sum(results[-5:])/5

def simulations_old(parent,children_values):
    """Heavy rollout vs light rollout 
    choosing children based on machine learning values """
    react = tanimoto_filtered_children(parent=parent,children=children_values)
    results =[]
    for i in range(0,len(children_values)):
        results.append(molecule_model_predict(children_values[i],parent)[0][1])
    rollout_reward = reward_dict(children_values,results)
    temp = []
    for item in rollout_reward.keys():
        temp.append([item,rollout_reward[item]])
    s = np.array(temp)
    return s, np.sum(results[-5:])/5, temp

def simulation__children_trees(parent,parent_trees):
    parent_trees[parent] = node(parent,0,0,0,0)
    for candidate in parent_trees[parent].children:
        parent_trees[candidate] = node(parent,0,0,0,0)
    return parent_trees

def tanimoto(initial,all_products): 
    """
    This function generates tanimoto distance between the parent and candidates. 
    Input type: Smile String, Smile String 
    Return type: Numerical similarity score
    """
    similarity_inital = []
    from rdkit.Chem import MACCSkeys
    for i in range(0,len(all_products)):
        mols = Chem.MolFromSmiles(initial),Chem.MolFromSmiles(all_products[i])
        fps = [ MACCSkeys.GenMACCSKeys(x) for x in mols ]
        similarity_inital.append([Chem.MolToSmiles(mols[1]),round(DataStructs.TanimotoSimilarity(fps[0], fps[1]), 4)])
    return similarity_inital

def molecule_model_predict(react,product):
    """
    product refers to the initial product and react refers to the HPPS elements. 
    """
    percentages = []
    input_vec = np.array(encoding(react)).reshape(1,2048)
    merged_reaction = ">>".join([react,product])
    individual_features = new_atomic_features(merged_reaction)
    temp_features = np.concatenate([individual_features[0],individual_features[1]])
    input_vec = np.concatenate([input_vec[0],temp_features]).reshape(1,2062)
    res = molecule_model.predict(input_vec)
    percentages.append([int(i),res[0][0],react]) 
    return percentages

def tanimoto_filtered_children(children,parent,threshold=0.3):
    df_similarity_inital = pd.DataFrame(tanimoto(parent,children),columns = ["Products","Similarity"])
    tanimoto_filtering = df_similarity_inital[df_similarity_inital["Similarity"]>threshold]
    de_duplicated_smiles = list(tanimoto_filtering["Products"])
    return de_duplicated_smiles

def searching_state_space(possible_reactants,pattern):
    reactions = np.array(possible_reactants)
    found_it = []
    positions = []
    for i in range(0,reactions.shape[0]):
        m = Chem.MolFromSmiles(pattern)
        m = Chem.AddHs(m)
        patt = Chem.MolFromSmiles(reactions[i])
        patt = Chem.AddHs(patt)
        if patt.HasSubstructMatch(m) and m.HasSubstructMatch(patt):
            found_it.append(patt)
            positions.append(i)
    return found_it, positions, len(positions)>=1


def get_exploration_values(first_children,threshold =0.7):
    values = []
    for i in range(0,len(first_children)):
        emp = reaction_model_predict(first_children[i])[0]
        values.append([emp[2],emp[1]])
    new = pd.DataFrame(values, columns = ["Smiles","Percentage"]).sort_values(by = "Percentage",ascending = False)
    top_results = new.iloc[:5,]["Smiles"]
    preferred = new[new["Percentage"] > threshold].shape[0]
    total = new.shape[0]
    return top_results, preferred, total

def simulation_run(child,depth):
    """
    Input:- [Name of parent for which simulation is being run]
    depth:- How many times do you want to generate children,(higher the value 
    beter the result)
    This function would generate children and use subsequent children as parent 
    and generate more candidates.
    """
    temp = [child]
    all_children = []
    new = []
    while depth:
        for i in range(0,len(temp)):
            value = generate_children(temp[i])
            all_children.extend(value)
        temp = all_children
        depth-=1 
    return all_children

def getting_list_of_generated_children(parent=SmilesBDO14,maximum_depth=1):
    first_children = generate_children(parent)
    list_of_children = []
    for i in range(0,len(first_children)):
        list_of_children.append(simulation_run(first_children[i],depth=maximum_depth))
    return first_children,list_of_children

def getting_mu_value(list_of_children,product):
    """This funcition generates the mu value for the list of generated candidates children 
    Input:- List of children, the parent candidate from which transformation was made.
    """
    mu_dict = {}
    va = []
    for i in range(0,len(list_of_children)):
        va.append(molecule_model_predict(list_of_children[i],product)[0][1])
    return np.max(va)

def running_mu_value(list_of_children,first_children):
    value_dict = {}
    for j in range(0,len(first_children)):
        value = getting_mu_value(list_of_children[j],first_children[j])
        value_dict[first_children[j]] = value
    return value_dict


def running_simulations(parent,simulation_threshold,depth = 2):
    top_results = [parent]
    top_results= 0
    preferred = 0
    """
    for i in range(0,len(top_results)):
        new.append(generate_children(top_results[i]))
    top_results, preferred, total = get_exploration_values(new[0])
    total+=total 
    preferred+= preferred
    print("iteration")"""
    temporary = []
    while depth:
        print("iteration")
        for i in range(0,len(top_results)):
            top_results, preferred, total = get_exploration_values(generate_children(top_results[i]),simulation_threshold)
            total+=total 
            preferred+= preferred
            temporary.append(top_results)
            print([preferred,total])
        top_results = temporary
        depth-=1
    return preferred, total




    