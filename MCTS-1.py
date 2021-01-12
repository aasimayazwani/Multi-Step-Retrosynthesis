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
    """Heavy rollout vs light rollout state_space_generation
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
    parent_trees[parent] = node(parent,0,0)
    for candidate in parent_trees[parent].children:
        parent_trees[candidate] = node(candidate,0,0)    
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

class Node :
    def __init__ (self , state , parent =None , statistics ={}):
        self.state = state
        self.parent = parent
        self.children = {}
        self.statistics = statistics

    def expand (self , action , next_state ):
        child = Node ( next_state , parent = self )
        self . children [ action ] = child
        return child

class MCTS_with_UCT :
    def __init__ (self , state , gamerules , C =1):
        self.game = gamerules
        self.C = C
        self.root = Node ( state , statistics ={" visits ": 0 ," reward ": np. zeros ( self . game . num_players ())})

     def is_fully_expanded (self , node ):
         return len ( self . game . get_actions ( node . state )) == len ( list ( node . children ))


     def best_action (self , node ):
        children = list ( node . children . values ())
        visits = np. array (([ child . statistics [" visits "] for child in children ]))
        rewards = np. array (([ child . statistics [" reward "] for child in children ]))
        total_rollouts = node . statistics [" visits "]
        pid = self . game . get_current_player_id ( node . state )
        # calculate UCB1 value for all child nodes
        ucb = ( rewards [: , pid ]/ visits + self . C *np. sqrt (2* np. log ( total_rollouts )/ visits ))
        best_ucb_ind = np. random . choice (np. flatnonzero ( ucb == ucb . max ()))
        return list ( node . children . keys ())[ best_ucb_ind ]

    def state_space_generation(molefile,unique_strings,hydrogen_add=True):
        item = []
        indexes = []
        rules = []
        from rdkit.Chem import rdChemReactions
        for i in range(0,len(unique_strings)):
            try:
                rxn = rdChemReactions.ReactionFromSmarts(unique_strings[i])
                if hydrogen_add == True:
                    reacts = Chem.AddHs(Chem.MolFromSmiles(molefile))
                if hydrogen_add == False:
                    reacts = Chem.MolFromSmiles(molefile)
                products = rxn.RunReactants((reacts,))
                for j in range(0,len(products)):
                    try:
                        if products != ():
                            Chem.SanitizeMol(products[j][0])
                            temp = Chem.AddHs(products[j][0])
                            item.append([i,Chem.MolToSmiles(temp)])
                    except:
                        pass
            except:
                pass
        if len(item)<1:
            #print("No possible state generated increase rule size, f= state_space_generation")
            return []
        else:
            reactions = pd.DataFrame(item)
            reactions.columns = ["rulenumber","products"]
            return reactions
