from .petri_nets import *


class ColoredPetriNet(PetriNet):
    def __init__(self, name):
        super().__init__(name)
        
    def fire_transition(self, transition_name, token_type='default'):
        if transition_name in self.transitions.keys():
            if self.transitions[transition_name].in_token_type == token_type:
                flag = self.transitions[transition_name].fire(token_type)
                self.update_net()
                return flag
            else:
                print('firing non-in-type token')
                flag = self.transitions[transition_name].fire(self.transitions[transition_name].in_token_type)
                self.update_net()
                return flag
        else:
            print('No such transition')
            return False
        
    def update_net(self):
        for t in self.transitions.values():
            t.check_firability()
            
    def observe(self):
        state = []
        for p in self.places.values():
            tmp = []
            for t in p.token.values():
                tmp.append(t)
            state.append(tmp)
        max_width = max(len(sublist) for sublist in state)
        new_state = [sublist + [0] * (max_width - len(sublist)) for sublist in state]
        return np.array(new_state).transpose().flatten()
    
    @property
    def observation_space(self):
        self.update_net()
        return self.observe().shape