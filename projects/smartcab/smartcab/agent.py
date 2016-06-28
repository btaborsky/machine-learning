import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.actions =  [None, 'forward', 'left', 'right']
        self.state_q_dict = {}
        
        
        self.learning_rate = 0.9
        self.discount_rate = 0.1
        self.random_prob_start = .2



        self.random_prob = self.random_prob_start
        self.cumulative_reward = 0
        self.cumulative_penalties = 0
        self.successes = []
        self.all_times_remaining = []
        self.time_remaining = None
        
        self.currentSuccess = False
        
        
    

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        # if self.currentSuccess:
        #     print "adding 1 to successes"
        # else:
        #     print "adding 0 to successes"

        self.successes.append(1 if self.currentSuccess else 0)
        self.all_times_remaining.append(self.time_remaining if self.currentSuccess else 0)       
        self.currentSuccess = False
        #make self.random_prob go down over time to a very small amount, to end up at 0 at trial #75

        self.random_prob = -len(self.successes)*self.random_prob_start/75 + self.random_prob_start
        

        print "Total successes: {}, Cumulative Reward: {}, Last failure: {}".format(np.sum(self.successes),self.cumulative_reward,self.find_last_failure())

        
    def find_last_failure(self):
        if 0 in self.successes:
            return (len(self.successes) - self.successes[::-1].index(0))-1
        return -1

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (self.next_waypoint, inputs['light'],inputs['oncoming'],inputs['right'],inputs['left'])

        if self.state not in self.state_q_dict:
            
            self.state_q_dict[self.state] = {}

            #this initialization works great but seems like it's kind of cheating
            # for act in self.actions:
            #     if act == None:
            #         new_val = random.random()-1
            #     else:
            #         new_val = random.random()
            #     self.state_q_dict[self.state][act] = new_val

            #this puts everything in an equal playing field with starting Q values between -.5 and .5
            for act in self.actions:
                self.state_q_dict[self.state][act] = random.random()-.5



        action_q_dict = self.state_q_dict[self.state]



        
        # TODO: Select action according to your policy
        action = max(action_q_dict.keys(),key = lambda x: action_q_dict[x])

        if random.random() < self.random_prob:
            action = random.choice(self.actions)

       

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.cumulative_reward += reward
        if reward < 0:
            self.cumulative_penalties += reward

        if reward >= 3.:
            #print "hello"
            
            self.currentSuccess = True
            self.time_remaining = deadline
            #print "Success achieved!"


        # TODO: Learn policy based on state, action, reward
        max_q = self.get_max_q()
        update = self.learning_rate*(reward + self.discount_rate*max_q)
        self.state_q_dict[self.state][action] = update + (1-self.learning_rate)*self.state_q_dict[self.state][action]

        #print "LearningAgent.update(): next_waypoint = {}".format(self.next_waypoint)

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def get_max_q(self):
        all_rewards = []
        for state in self.state_q_dict:
            for action in self.state_q_dict[state]:
                all_rewards.append(self.state_q_dict[state][action])
        return max(all_rewards)



def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    num_successes = np.sum(a.successes)
    last_failure = a.find_last_failure()
    total_penalty = a.cumulative_penalties
    avg_time_remaining = np.mean(a.all_times_remaining)

    print "Total number of successes: {}".format(num_successes)
    print "Failure last occurred at trial: {}".format(last_failure)
    print 'Total penalties incurred: {}'.format(total_penalty)
    print "Average time remaining: {}".format(avg_time_remaining)


    for state in a.state_q_dict:
        print state
        for action in a.state_q_dict[state]:
            print "Action: {}, Q: {:2f}".format(action,a.state_q_dict[state][action])

    print a.state_q_dict[('right','red',None,None,None)]
    
    return (num_successes,last_failure,total_penalty,avg_time_remaining)



if __name__ == '__main__':
    
    #lists for results over multiple runs
    # all_avg_time_remaining = []
    # all_num_successes = []
    # all_last_failures = []
    # all_total_penalties = []    
    # for i in range(20):
    #     (n_successes,last_failure,total_penalty,avg_time_remaining) = run()
    #     all_num_successes.append(n_successes)
    #     all_last_failures.append(last_failure)
    #     all_total_penalties.append(total_penalty)
    #     all_avg_time_remaining.append(avg_time_remaining)
    # print "Num successes mean: {}".format(np.mean(all_num_successes))
    # print "Last failures mean: {}".format(np.mean(all_last_failures))
    # print "total penalties mean: {}".format(np.mean(all_total_penalties))
    # print "Avg times remaining mean: {}".format(np.mean(all_avg_time_remaining))

    results = run()
