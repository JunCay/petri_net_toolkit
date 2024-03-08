import pandas as pd
import numpy as np
import random
import math

class Element():
    def __init__(self, name, note=''):
        self.name = name
        self.note = note
        self.ins = set()
        self.outs = set()
        
    def set_note(self, new_note=''):
        self.note = new_note
        
    def get_note(self):
        print(self.name, ': ', self.note)
        
        
class Place(Element):
    def __init__(self, name, note='', type='non-lock', activity_type='activity'):
        super().__init__(name, note)
        self.token = {'default':0}
        self.place_type = type                  # non-lock, lock 
        self.activity_type = activity_type      # activity, resource, operation, idle
    
    def add_token(self, token_type='default'):
        # if token_type not in self.token.keys():
        #     return False
        self.token[token_type] += 1
        return True
    
    def take_token(self, token_type='default'):
        if token_type not in self.token.keys():
            print('Unexpected token type')
            return False
        if self.token[token_type] <=0:
            print('Unexpected take action')
            return False
        self.token[token_type] -= 1
        
    def set_token(self, token_num, token_type='default'):
        if token_type in self.token.keys():
            self.token[token_type] = token_num
            return False        # old key
        self.token[token_type] = token_num
        return True             # new key
        
    def get_token(self, token_type='default'):
        print('Token ', token_type,' : ', self.token)
        return self.token[token_type]
    
        
class Transition(Element):
    def __init__(self, name, note='', type='auto', time=0, reward=0, dist_type='gaussian',mu=0, sigma=0, in_token_type='default', out_token_type='default', error_token_type='default_error', cv=0.0, pass_type='one'):
        super().__init__(name, note)
        self.trans_type = type      # auto(default as Immediate), timed, stochastic, macro
        self.pass_type = pass_type  # one(to send selected token), all(to send all token)
        self.time0 = time
        self.reward = reward
        self.dist_type = dist_type
        self.in_token_type = in_token_type
        self.out_token_type = out_token_type
        self.error_token_type = error_token_type
        self.cv = cv            # coefficient of variation
        self.mu = mu
        self.sigma = sigma
        self.check_firability()
        
    @property
    def time(self):
        if self.trans_type == 'stochastic':
            if self.dist_type == 'uniform':
                return random.uniform(self.mu-math.sqrt(3)*self.sigma, self.mu-math.sqrt(3)*self.sigma)
            else:
                return random.gauss(mu=self.mu, sigma=self.sigma) 
        return self.time0
        
    def check_firability(self, token_type='default'):
        for p in self.outs:
            if p.place_type == 'lock':
                if p.token[token_type] >= 1:
                    self.state = -1
                    return False
        if self.pass_type == 'one':
            for p in self.ins:
                if not self.in_token_type in p.token.keys():
                    self.state = -1
                    return False
                if p.token[self.in_token_type] <= 0:
                    self.state = -1
                    return False
            self.state = 1
            return True
        elif self.pass_type == 'all':
            for p in self.ins:
                flag = False
                for tk in p.token.values():
                    if tk > 0:
                        flag = True
                        break
                if flag == False:
                    self.state = -1
                    return False
            self.state = 1
            return True
        else:
            print('Unknown pass type')
            self.staet = -1
            return False
    
    def fire(self, token_type='default'):
        if self.pass_type == 'one':
            if self.check_firability(self.in_token_type):
                for p in self.ins:
                    # print(self.ins)
                    # print(p.name, 'at ',p, 'token token, now remain: ', p.token['default'])
                    p.take_token(self.in_token_type)

                cv = random.random()        # Coefficient of variation
                if cv >= self.cv:
                    for p in self.outs:
                        p.add_token(self.out_token_type)
                    return True
                else:
                    for p in self.outs:
                        p.add_token(self.error_token_type)
                    return True
            else:
                # print(self.name, ' Unfirable')
                return False
        elif self.pass_type == 'all':           # 'all' is for transport, 1 out = 1 in
            if self.check_firability():
                token_token_type = 'default'
                for p in self.ins:
                    if p.activity_type != 'resource':
                        for tk_k in p.token.keys():
                            if p.token[tk_k] > 0:
                                p.token[tk_k] -= 1
                                token_token_type = tk_k
                                # print('1', p.name, token_token_type)
                                break
                    
                cv = random.random()        # Coefficient of variation
                if cv >= self.cv:
                    for p in self.outs:
                        # print(p.name, p.token)
                        # print('2', token_token_type)
                        if p.activity_type == 'resource':
                            p.add_token()
                        else:
                            p.add_token(token_type=token_token_type)
                        # print(p.name, p.token)
                    return True
                else:
                    for p in self.outs:
                        p.add_token(self.error_token_type)
                    return True
            else:
                # print(self.name, ' Unfirable')
                return False
                            
                
        