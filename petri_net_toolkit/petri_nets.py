from .elements import *
import pygraphviz as pgv
from IPython.display import Image
import collections
import copy

class PetriNet():
    def __init__(self, name):
        self.name = name
        self.places = dict()
        self.transitions = dict()
        self.pt_matrix = pd.DataFrame(columns=['default_c'], index=['default_r'])
        self.initial_dict = dict()
        self.act_time = 0
        self.invalid_fire_penalty = -1
        self.debugger_num = 0
        
    def set_invalid_fire_penalty(self, penalty):
        self.invalid_fire_penalty = penalty
        
    def add(self, element):
        if isinstance(element, Place):
            if element in self.places.values():
                print('Place already added')
                return False
            self.places[element.name] = element
            self.pt_matrix.loc[element.name] = np.zeros(len(self.pt_matrix.columns))
            return True
        elif isinstance(element, Transition):
            if element in self.transitions.values():
                print('Transition already added')
            self.transitions[element.name] = element
            self.pt_matrix[element.name] = 0
            return True
        self.update_net()
        return False
    
    def set_token(self, place_name, token_num, token_type='default'):
        return self.places[place_name].set_token(token_num, token_type)
        
    def add_token(self, place_name, token_type='default'):
        return self.places[place_name].add_token(token_type)
        
    def take_token(self, place_name, token_type='default'):
        return self.places[place_name].take_token(token_type)
        
    def get_token(self, place_name, token_type='default'):
        return self.places[place_name].get_token(token_type)
    
    def fire_transition(self, transition_name, token_type='default'):
        if transition_name in self.transitions.keys():
            flag = self.transitions[transition_name].fire(token_type)
            self.update_net()
            return flag
        else:
            print('No such transition')
            return False
    
    def initialize_net(self):
        self.state_dim = self.observation_space
        self.action_dim = self.action_space
    
    def update_net(self):
        for t in self.transitions.values():
            t.check_firability(token_type='default')
            
    def update_auto_transitions(self):
        reward = 0
        for t in self.transitions.values():
            if t.trans_type == 'auto':
                while t.check_firability():
                    t.fire()
                    self.act_time += t.time
                    reward += t.reward * self.time_decay(t.time)
        self.update_net()
        return reward
        
    def define_initial(self, initial_dict):
        if len(initial_dict) > len(self.places):
            print('Invalid dict length')
            return False
        self.initial_dict=initial_dict
        return True
        
    def check_alive(self):
        for t in self.transitions.values():
            if t.check_firability():
                # print(t.name, 'alive')
                return True
        return False
        
    # environment tools
    def reset(self):
        self.act_time = 0
        if not self.initial_dict:
            print('Define initial dictionary first')
            return None
        for p in self.places.values():
            for t in p.token.keys():
                p.token[t] = 0
        for p_name in self.initial_dict.keys():
            self.places[p_name].token = copy.deepcopy(self.initial_dict[p_name])
        
        return self.observe()
    
    def time_decay(self, time, mean_time=20, normalizer=1):
        penalty = 1 / (1 + np.exp(time-mean_time))        ## Time penalty normalization 
        penalty *= normalizer
        return penalty
    
    
    def get_invalid_fire_penalty(self):
        return self.invalid_fire_penalty
    
    def time_penalty(self, time):
        return -time
    
    def step(self, action):
        reward = 0
        
        if isinstance(action, int):
        # ==== int action ====
            if action > self.action_dim:
                print('Invalid Action Dimension')
                return False
            
            if self.fire_transition(self.actions[action].name):
                self.act_time += self.actions[action].time
                reward += self.actions[action].reward * self.time_decay(self.actions[action].time, mean_time=self.actions[action].mu)
                reward += self.time_penalty(self.actions[action].time)
                # print('get reward: ', reward)
            else:
                reward += self.get_invalid_fire_penalty()
            
            reward += self.update_auto_transitions()
            # print('get penalty: ', reward)
        
        # ==== vector action ====
        else: 
            action = np.array(action)            
            if action.shape[0] != self.action_dim:
                print('Invalid Action Dimension (vector)')
                return False
            for i in range(len(action)):
                max_time = 0
                if action[i] == 1:
                    if self.fire_transition(self.actions[i].name):
                        # print(self.actions[i].name, ' fired')
                        reward += 1     # fire bonus
                        reward += self.actions[i].reward  * self.time_decay(self.actions[i].time, mean_time=self.actions[i].mu)
                        if self.actions[i].time > max_time:
                            max_time = self.actions[i].time
                        reward += self.time_penalty(self.actions[i].time)
                    else:
                        reward += self.get_invalid_fire_penalty() * 0.1
                    self.act_time += max_time
                    reward += self.update_auto_transitions()
        next_state = self.observe()
        done = not self.check_alive()
        
        # if done:
        #     self.debugger_num+=1
        #     name = 'debugger_pics/debugger' + str(self.debugger_num) + '.png'
        #     self.draw_net(name)
        # print(self.debugger_num, done)
        # print(next_state)
        return next_state, reward, done
    
    @property
    def observation_space(self):
        self.update_net()
        lt = 1
        for p in self.places.values():
            if len(p.token) > lt:
                lt = len(p.token)
        return len(self.places), lt
    
    def observe(self):
        # state = []
        # for p in self.places.values():
        #     row = []
        #     for t in p.token.values():
        #         row.append(t)
        #     state.append(row)
        # return np.array(state)        ## 待优化颜色petri
        state = []
        for p in self.places.values():
            state.append(p.token['default'])
        return np.array(state)
    
    ## state normalization
    
    @property
    def action_space(self):
        self.update_net()
        action_dim = 0
        self.actions = []
        for t in self.transitions.values():
            if t.trans_type == 'timed' or t.trans_type == 'stochastic':
                action_dim += 1
                self.actions.append(t)          # action_list 按序号存储可执行transition
        return action_dim
    
    # building tools    
    def link(self,element1, element2):
        if isinstance(element1, Place):
            if element1 not in self.places.values():
                print(f'Add place {element1.name} first')
                return False
            if not isinstance(element2, Transition):
                print('Unexpected pair type')
                return False
            self.places[element1.name].outs.add(element2)
            self.transitions[element2.name].ins.add(element1)
            self.pt_matrix.loc[element1.name, element2.name] = -1
            return True
            
        elif isinstance(element1, Transition):
            if element1 not in self.transitions.values():
                print('Add this place first')
                return False
            if not isinstance(element2, Place):
                print('Unexpected pair type')
                return False
            self.places[element2.name].ins.add(element1)
            self.transitions[element1.name].outs.add(element2)
            self.pt_matrix.loc[element2.name, element1.name] = 1
            return True
        
        print('Unexpected pair type')
        return False
    
    # plotting tools
    def draw_net(self, chart_name='default_pic_name.png', show_note=False):
        self.update_net()

        G = pgv.AGraph(directed=True)
        for place in self.places.values():
            G.add_node(place.name)
        for transition in self.transitions.values():
            G.add_node(transition.name)            
        for index, row in self.pt_matrix.iterrows():
            for col in self.pt_matrix.columns:
                link = row[col]
                if link == 1:       # into place
                    G.add_edge(col, index)
                    G.get_edge(col, index).attr['style'] = 'solid'
                elif link == -1:
                    G.add_edge(index, col)
                    G.get_edge(index, col).attr['style'] = 'solid'
        
        for place in self.places.values():
            G.get_node(place.name).attr['shape'] = 'circle'
            G.get_node(place.name).attr['label'] = place.name + ': ' + str(place.token)
            if place.place_type == 'lock':
                G.get_node(place.name).attr['color'] = 'red'
            if show_note:
                G.get_node(place.name).attr['label'] += ('\n# ' + place.note)
        for transition in self.transitions.values():
            G.get_node(transition.name).attr['shape'] = 'box'
            if transition.state == -1:
                G.get_node(transition.name).attr['color'] = 'red'
            elif transition.state == 1:
                G.get_node(transition.name).attr['color'] = 'blue'
            if show_note:
                G.get_node(transition.name).attr['label'] += ('\n# ' + transition.note)

        G.draw(chart_name, prog='dot', format='png')
        Image(filename=chart_name)
    
    def print_pt_matrix(self):
        print(self.pt_matrix)
        
    def print_place_tokens(self):
        print(f'At time {self.act_time}, the state is')
        for p in self.places.values():
            print(p.name, ': ', p.token) 
        print('\n')
    
    
class SubPetriNet(PetriNet, Transition):
    def __init__(self, name, net:PetriNet, parent_trans:Transition):
        self.name = name
        self.places = net.places            # copy.deepcopy(net.places)
        self.transitions = net.transitions  # copy.deepcopy(net.transitions)
        self.pt_matrix = net.pt_matrix      # copy.deepcopy(net.pt_matrix)
        self.trans_type = 'Macro'
        self.ins = parent_trans.ins
        self.outs = parent_trans.outs
        self.time = self.get_net_time()
        self.check_firability()
        
    def get_net_time(self):
        time = 0
        return time
    
    def check_firability(self, token_type='default'):
        for t in self.transitions.values():
            if not t.check_firability():
                return False
        return True
    
    def fire(self, token_type='default'):
        if super().fire(token_type):
            
            return True
        return False
    

