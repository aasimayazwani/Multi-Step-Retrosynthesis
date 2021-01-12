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
