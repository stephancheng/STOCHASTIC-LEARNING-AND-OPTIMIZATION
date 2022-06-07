#https://python-advanced.quantecon.org/discrete_dp.html
# import quantecon as qe
# import scipy.sparse as sparse
# from quantecon import compute_fixed_point
from quantecon.markov import DiscreteDP
from quantecon.markov.ddp import backward_induction
from scipy.stats import betabinom
import numpy as np

class Room:

    def __init__(self, α=0.5, β=0.9, state_space = [2, 7], A = 3, price = 10, discount = 0.5):
        """
        Set up R, Q and β, the three elements that define an instance of
        the DiscreteDP class.
        """

        self.n = np.prod(state_space) # number of state
        self.m = A # number of action
        self.p = price
        self.d = discount
        self.day = state_space[1] # number of day for long stay customers

        self.update_prob(α, β)
        self.populate_R()
        self.populate_action_space()

    def prob_mean(self):
        self.prob = self.α/ (self.α + self.β)

    def update_prob(self, α, β):
        self.α , self.β = α, β
        self.prob_mean()
        self.populate_Q()

    def populate_R(self):
        """
        Populate the R matrix, with R[s, a] = -np.inf for infeasible
        state-action pairs.
        """
        self.Reward = np.zeros((self.n, self.m))
        self.R = []
        for s in range(self.n):
            if s == self.day:
                self.R = self.R + [self.p, self.p * self.d / self.day, 0]
                
            elif s == 0:
                self.R = self.R + [self.p * self.d / self.day, 0]
            else:
                self.R = self.R + [self.p * self.d / self.day]
                
        # index
        # 0 to self.day: X0; self.day to self.day*2 - 1:x1

        #h = 0 & a =a1: rt= p
        self.Reward[0, 0] = self.p
        self.Reward[self.day, 0] = self.p

        #h = 0 & a = a3: rt=0
        self.Reward[0, 2] = 0
        self.Reward[self.day, 2] = 0

        #h > 1 or a= a2: rt= y  d7  100
        self.Reward[1:self.day, :] = self.p * self.d / self.day
        self.Reward[self.day +1:, :] = self.p * self.d / self.day
        self.Reward[0,1] = self.p * self.d / self.day
        self.Reward[self.day,1] = self.p * self.d / self.day

    def populate_Q(self):
        """
        Populate the Q matrix by setting

            Q[s, a, s'] = 1 / (1 + B) if a <= s' <= a + B

        and zero otherwise.
        """
        self.Q= []
        for s in range(self.n):
            if s == self.day:
                #a = 0
                q = np.zeros(self.n)
                q[0] = 1- self.prob
                q[self.day] = self.prob
                self.Q.append(q)
            if s == self.day or s == 0:
                # a = 1
                q = np.zeros(self.n)
                q[self.day - 1] = 1- self.prob
                q[self.day *2 -1] = self.prob
                self.Q.append(q)
                
                # a = 2
                q = np.zeros(self.n)
                q[0] = 1 - self.prob
                q[self.day] = self.prob
                self.Q.append(q)
                
            else:
                q = np.zeros(self.n)
                q[s%self.day - 1] = 1- self.prob
                q[s%self.day - 1 + self.day] = self.prob
                self.Q.append(q)
        '''
        self.Q = np.zeros((self.n, self.m, self.n))
        for s in range(self.n):
            if s == 0 or s == self.day:
                self.Q[s, 0, 0] = 1- self.prob # a =0
                self.Q[s, 1, self.day - 1] = 1- self.prob # a = 1
                self.Q[s, 2, 0] = 1- self.prob # a = 2

                self.Q[s, 0, self.day] = self.prob # a = 0
                self.Q[s, 1, self.day *2 -1] = self.prob # a = 1
                self.Q[s, 2, self.day] = self.prob # a = 2
            else:
                # if 0<h<num_day, next state = h-1 for all a
                
                self.Q[s, :, s%self.day - 1] = 1- self.prob
                self.Q[s, :, s%self.day - 1 + self.day] = self.prob
                '''
    def populate_action_space(self):
        self.s_indices = []
        self.a_indices = []

        for s in range(self.n):
            if s == self.day:
                self.s_indices = self.s_indices + [s, s, s]
                self.a_indices = self.a_indices + [0,1,2]
            elif s == 0:
                self.s_indices = self.s_indices + [s, s]
                self.a_indices = self.a_indices + [1,2]
            else:
                self.s_indices = self.s_indices + [s]
                self.a_indices = self.a_indices + [2]

def solve_dp(aroom, discount_factor = 0.99, T = None):
    R, Q, s_indices, a_indices = aroom.R, aroom.Q, aroom.s_indices, aroom.a_indices
    ddp = DiscreteDP(R, Q, discount_factor, s_indices, a_indices)

    if discount_factor == 1:
        res = backward_induction(ddp, T)
        res = res[1][0]
    else:
        res = ddp.solve(method='policy_iteration', v_init=[0] * aroom.n,\
                    epsilon=0.01)
        res = res.sigma
    return res

def compute_action(aroom, state, T):
    res = solve_dp(aroom, discount_factor = 1, T = T)
    discrete_state = aroom.day * state[0] + state[1]
    return discrete_state, res[discrete_state]

def step(state, cus, action, days):
    h = state[1]
    if h > 0:
        return [cus, h - 1]
    elif action == 1:
        return [cus, days - 1]
    else:
        return [cus, 0]

def run(α=0.1, β=0.9, α_=0.1, β_=0.9, state_space = [2, 7], A = 3, price = 10, discount = 0.7, T= 182):
    
    aroom = Room(α_,β_, state_space, A, price, discount)
    customers = betabinom.rvs(1, α, β, size = T) # Random T days customers
    state, action = None, None
    rewards = []
    
    for i, cus in enumerate(customers):
        # TODO
        #α_,β_= update_believe(α_, β_, cus)
        aroom.update_prob(α, β)
        if not state:
            state = [cus, 0]
        else:
            state = step(state, cus, action, aroom.day)
        discrete_state, action = compute_action(aroom, state, T-i)
        reward = aroom.Reward[discrete_state, action]
        print("state:{}, action:{}, reward:{}".format(state, action, reward))
        rewards.append(reward)
        
    # TODO: Myopic policy & reward
    
    return rewards

# TODO: RUN 1000 stimulations with random αβ and believe
rewards = run()
print(sum(rewards))