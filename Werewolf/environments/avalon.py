from chatarena.environments.base import TimeStep, Environment
from chatarena.message import Message, MessagePool
from chatarena.utils import extract_jsons
from typing import List, Union, Dict
from abc import abstractmethod

from ..message import Message
from ..config import EnvironmentConfig

import random
random.seed(42)
import numpy as np

import json
import pdb
from ast import literal_eval
from collections import defaultdict

TEAM_GOOD = ["Merlin", "Percival", "Loyal Servant of Arthur"]
TEAM_EVIL = ["Mordred", "Morgana", "Assassin", "Oberon"]
MAX_FAILED_CONSECUTIVE_VOTES = 2

class Avalon(Environment):
    type_name = "avalon"

    def __init__(self, player_names: dict, quest_sizes: list, **kwargs):
        super().__init__(player_names=player_names, quest_sizes=quest_sizes, **kwargs)
        self.player_names = player_names
        self.anonymous_player_names = [f"Player {i}" for i in range(len(player_names))]
        self.quest_sizes = quest_sizes

        self.num_discussed_players = 0 # used for tracking # players who have participated in the current discussion round
        self.num_discussions = 0  # max 3 rounds for team building

        self.success_quests = 0
        self.failed_quests = 0
        self.num_voted_players = 0
        self.last_vote_success = None
        self.failed_consecutive_votes = 0
        self.assassinated_player = None
        self.message_pool = MessagePool()

        self.current_phase = "team_building"
        self.current_turn = 0
        self.current_vote_round = 0
        self.current_quest_idx = 0
        self.current_leader_idx = 0 # random.choice(range(len(self.player_names)))
        self.current_player_idx = self.current_leader_idx
        self.current_players_on_quest = []
        self.merlin_idx = None
        self.percival_idx = None
        self.current_player_votes = {}
        self.current_quest_votes = defaultdict(int)
        self.all_players_votes = []
        self.all_quests_votes = []

        self._terminal = False
        self._initialized = False
        self._changed_leader = False

        self.reset()

    def _moderator_speak(self, text: str, visible_to: Union[str, List[str]] = "all"):
        """
        moderator say something to inform the game status and collect quest results
        """
        message = Message(agent_name="Moderator", content=text, turn=self.current_turn, visible_to=visible_to)
        self.message_pool.append_message(message)

    def _get_player_info(self):
        evil_ids, merlin_ids, percival_ids = [], [], []
        print(self.player_names)
        for i, name in enumerate(self.player_names):
            if name in ["Mordred", "Morgana", "Assassin"]:
                evil_ids.append(i)
            if name in ["Mordred", "Morgana", "Assassin", "Oberon"]:
                merlin_ids.append(i)
            if name in ["Merlin", "Morgana"]:
                percival_ids.append(i)
        return {"evil": evil_ids, 
                "merlin": merlin_ids, 
                "percival": percival_ids}

    def _initiate_votes(self):
        self.current_player_votes = {player: 0 for player in self.anonymous_player_names}
    
    def _initiate_current_votes(self):
        self.current_quest_votes = {player: 0 for player in self.current_players_on_quest}

    @abstractmethod
    def reset(self):
        self.num_discussed_players = 0 # used for tracking # players who have participated in the current discussion round
        self.num_discussions = 0  # max 3 rounds for team building
        
        self.success_quests = 0
        self.failed_quests = 0
        self.num_voted_players = 0
        self.last_vote_success = None
        self.failed_consecutive_votes = 0
        self.assassinated_player = None
        self.message_pool.reset()

        self.current_phase = "team_building"
        self.current_turn = 0
        self.current_vote_round = 0
        self.current_quest_idx = 0
        self.current_leader_idx = 0 # random.choice(range(len(self.player_names)))
        self.current_player_idx = self.current_leader_idx
        self.current_players_on_quest = [] #['Player 0', 'Player 3']
        self.merlin_idx = None
        self.percival_idx = None
        self.current_player_votes = {}
        self.current_quest_votes = defaultdict(int)
        self.all_players_votes = []
        self.all_quests_votes = []

        self._terminal = False
        self._initialized = False
        self._changed_leader = False
        self._initiate_votes()

        # Moderator declares the start of game
        player_info = self._get_player_info()
        print(player_info)

        self._merlin_idx = [i for i, x in enumerate(self.player_names) if x == "Merlin"][0]
        self._percival_idx = [i for i, x in enumerate(self.player_names) if x == "Percival"][0]

        self._moderator_speak(f"This game will consist of {self.num_players} players. I will now privately reveal your roles, and provide you with any additional information according to your role.")

        for i,x in enumerate(self.player_names):
            mod_text = f"Player {i}, your role is {x}."
            if x == 'Morgana' or x == 'Assassin':
                evil_text = [f"Player {d}" for d in  player_info['evil']]
                mod_text += f" There are two agents of Evil (including you): {evil_text}."
            elif x == "Merlin":
                evil_text = [f"Player {d}" for d in  player_info['evil']]
                mod_text += f" There are two agents of Evil: {evil_text}. As Merlin, you must be careful about revealing your identity. If the Assassin can correctly guess you to be Merlin, you will lose the game."
            elif x == "Percival":
                percival_text = "Player {0} and Player {1}".format(player_info['percival'][0], player_info['percival'][1])
                mod_text += f" As Percival, your job is to identify Merlin and carefully support him. Of the following players, one is Morgana (pretending to be Merlin) and the other is Merlin: {percival_text}"
            self._moderator_speak(mod_text, visible_to=f"Player {i}")
            print(i, mod_text)
        #self._moderator_speak(f"Minions of Mordred, these players {player_info['evil']} are all agents of Evil.", visible_to=["Morgana", "Assassin"])
        #self._moderator_speak(f"Merlin, these players {player_info['merlin']} are the agents of Evil.", visible_to="Merlin")
        #self._moderator_speak(f"Percival, these two players {player_info['percival']} are either Merlin or Morgana.", visible_to="Percival")
        #self._moderator_speak(f"Minions of Mordred, these players [{player_info['evil']}] are all agents of Evil.", visible_to=player_info['evil'])
        #self._moderator_speak(f"Merlin, these players [{player_info['merlin']}] are the agents of Evil.", visible_to=self._merlin_idx)
        #self._moderator_speak(f"Percival, these two players [{player_info['percival']}] are either Merlin or Morgana", visible_to=self._percival_idx)
        #self._moderator_speak(f"All players open your eyes. The first leader is Player {self.current_leader_idx}. Player {self.current_leader_idx}, please propose a team for the first quest. This quest requires {self.quest_sizes[0]} players to finish. You can and should propose yourself as a member of the quest team unless there is a strong reason not to do so. All players, there will be three rounds of discussion before you vote for the proposed team. Please make sure to use the discussions wisely.") 
        self._moderator_speak(f"All players now know their roles and all necessary information. The first leader is Player {self.current_leader_idx}. Player {self.current_leader_idx}, please propose a team for the first quest. This quest requires {self.quest_sizes[0]} players to finish. You can and should propose yourself as a member of the quest team unless there is a strong reason not to do so. All players, there will be three rounds of discussion before you vote for the proposed team. Please make sure to use the discussions wisely.") 
        self.message_pool.print()

        # Get next player name
        print(f"The current leader is: {self.anonymous_player_names[self.current_leader_idx]}")
        observation = self._get_observation(self.player_names[self.current_leader_idx])
        #print(observation)

        # Initiate votes
        self._players_votes = {name: 0 for name in self.anonymous_player_names}
        print("Initial votes:", self._players_votes)
        
        # Set initialized to True
        self._initialized = True
        #pdb.set_trace()

        return TimeStep(observation=observation, reward=self._get_zero_rewards(), terminal=False)

    def to_config(self) -> EnvironmentConfig:
        self._config_dict["env_type"] = self.type_name
        return EnvironmentConfig(**self._config_dict)

    @property
    def num_players(self) -> int:
        """
        get the number of players
        """
        return len(self.player_names)

    @abstractmethod
    def _get_next_player(self) -> int:
        """
        Return the name of the next player.

        Note:
            This method must be implemented by subclasses.

        Returns:
            int: The index of the next player.
        """
        if self._changed_leader:
            self.current_player_idx = self.current_leader_idx
            self._changed_leader = False
        elif not self._changed_leader and self.current_player_idx < len(self.player_names) - 1:
            self.current_player_idx += 1
        else:
            self.current_player_idx = 0
        #print(self.player_names[self.current_player_idx])
        return self.current_player_idx

    @abstractmethod
    def _get_next_leader(self) -> int:
        if self.current_leader_idx == len(self.player_names) - 1:
            self.current_leader_idx = 0
        else:
            self.current_leader_idx += 1
        self._changed_leader = True
        return self.current_leader_idx
    
    @abstractmethod
    def _get_next_quest(self) -> int:
        if self.current_quest_idx == len(self.quest_sizes) -1:
            self.current_quest_idx = 0
            print("All quests have been completed. Game has come to an end.")
        else:
            self.current_quest_idx += 1
        return self.current_quest_idx

    @abstractmethod
    def _get_observation(self, player_name=None) -> List[Message]:
        """
        Return observation for a given player.

        Note:
            This method must be implemented by subclasses.

        Parameters:
            player_name (str, optional): The name of the player for whom to get the observation.

        Returns:
            List[Message]: The observation for the player in the form of a list of messages.
        """
        if player_name is None:
            return self.message_pool.get_all_messages()
        else:
            return self.message_pool.get_visible_messages(player_name, turn=self.current_turn)

    @abstractmethod
    def print(self):
        """
        print the environment state
        """
        self.message_pool.print()
    
    def print_stage(self):
        print(f"Current game stage: {self.current_phase}, at discussion round {self.num_discussions}")

    @abstractmethod
    def _text2vote(self, text) -> str:
        """
        convert text to vote, return anonymous votes without revealing the identities of voters
        """
        try:
            d = literal_eval(text)
            if "vote" in d:
                d["vote"] = int(d["vote"])
                return 1 if d["vote"] == 1 else 0
            else:
                print("Wrong voting format!")
                print(text)
                return 0  
        except:
            print("Wrong voting format!")
            print(text)
            return 0

    def _get_players_on_quest(self, text: str) -> list:

        try:
            team = literal_eval(text)
            if "players_on_quest" in team:
                return team["players_on_quest"]
            else:
                print("Wrong team proposal format!")
                print(team)
                return []
        except:
            print("Wrong team proposal format!")
            print(text)
            return []

    def _get_assassinated_player(self, text: str) -> str:
        
        try:
            team = literal_eval(text)
            if "player_to_assassinate" in team:
                return team["player_to_assassinate"]
            else:
                print("Wrong assassination format!")
                print(team)
                return ""
        except:
            print("Wrong assassination format!")
            print(text)
            return ""


    @abstractmethod
    def step(self, player_name: str, action: str) -> TimeStep:
        """
        Execute a step in the environment given an action from a player.

        Note:
            This method must be implemented by subclasses.

        Parameters:
            player_name (str): The name of the player.
            action (str): The action that the player wants to take.

        Returns:
            TimeStep: An object of the TimeStep class containing the observation, reward, and done state.
        """
        assert self._initialized == True
        if not self._initialized:
            self.reset()
    
        player_name = self.anonymous_player_names[self.current_player_idx]
        assert player_name == f"Player {self.current_player_idx}", f"Wrong player! It is Player {self.current_player_idx}'s' turn."

        # Team building phase, maximum 3 rounds of discussions allowed before going into voting
        if self.current_phase == "team_building":
            if self.num_discussions < 3:
                message = Message(agent_name=player_name, content=action, turn=self.current_turn)
                self.message_pool.append_message(message)
                self.num_discussed_players += 1

                if self.num_discussed_players == len(self.player_names): 
                    self.num_discussions += 1
                    self.num_discussed_players = 0

            else:
                #print("There has been three rounds of discussion. ")
                #print(f"Total turns at team building: {self.current_turn}")
                self.current_phase = "team_voting"

                team_proposal_format = {'quest_idx': self.current_quest_idx, 'quest_leader': self.current_leader_idx, 'players_on_quest': [self.anonymous_player_names[self.current_leader_idx], self.anonymous_player_names[3]]}
                print(team_proposal_format)
                self._moderator_speak(f"Discussion is over. Player {self.current_leader_idx}, please propose your team in the following format: \n\nFor example: \n\n{json.dumps(team_proposal_format)}. Do not state anything else.")

                team_approve_format = {'player_idx': self.current_player_idx, 'vote': 1}
                team_disapprove_format = {'player_idx': self.current_player_idx, 'vote': 0}
                self._moderator_speak(f"All other players, it's time to vote for Player {self.current_leader_idx}'s proposed team. Announce your vote in the following format: \n\nIf you approve, say: \n\n{json.dumps(team_approve_format)} \n\notherwise say: \n\n{json.dumps(team_disapprove_format)}. Do not state anything else. If your vote does not follow the format, it will be discarded automatically.")

            self.current_turn += 1
            rewards = self._get_zero_rewards()
            observations = self._get_observation()
            print(f"Length of Observations: {len(observations)}")
            timestep = TimeStep(observation=observations,
                                reward=rewards,
                                terminal=False)  # Return all the messages

        elif self.current_phase == "team_voting":
            #print(f"there has been {self.num_discussions} rounds of discussion. Now propose a team.")
            if self.num_discussions == 3:
                self.num_discussions = 0

            message = Message(agent_name=player_name, content=action, turn=self.current_turn, visible_to="all")
            self.message_pool.append_message(message)

            if len(self.current_player_votes) < len(self.player_names):
                if self.current_player_idx == self.current_leader_idx:
                    self.current_players_on_quest = self._get_players_on_quest(action)
                    #assert len(self.current_players_on_quest) == self.quest_sizes[self.current_quest_idx]
                    self._moderator_speak(f"Current team proposal by Player {self.current_leader_idx}: {self.current_players_on_quest}")
                    self.current_player_votes[self.anonymous_player_names[self.current_leader_idx]] = 1 # the leader automatically approves the proposal
                else:
                    vote = self._text2vote(action)
                    self.current_player_votes[self.anonymous_player_names[self.current_player_idx]] = vote
            
            else:
                assert len(self.current_player_votes) == len(self.player_names)
                print(self.current_player_votes)
                if sum(self.current_player_votes.values()) > len(self.player_names) / 2:
                    self.last_vote_success = True
                    self._moderator_speak(f"This quest team has gotten majority votes. Now, all players on the quest can start this task.")
                    self.current_phase = "quest"
                else:
                    self.last_vote_success = False
                    self.current_leader_idx = self._get_next_leader()
                    self._moderator_speak(f"Not getting majority votes on the quest team proposal. Player {self.current_leader_idx} is the new leader. You can propose a new team for this quest.")
                    self.current_phase = "team_building"
                    #self.print_stage()
                    self.failed_consecutive_votes += 1
                    if self.failed_consecutive_votes >= MAX_FAILED_CONSECUTIVE_VOTES:
                        self.failed_quests += 1
                self.all_players_votes.append({"quest_idx": self.current_quest_idx,
                                                "quest_leader": self.current_leader_idx,
                                                "players_on_quest": self.current_players_on_quest,
                                                "voting_result": self.last_vote_success})
            
            self.current_turn += 1                
            rewards = self._get_zero_rewards()
            observations = self._get_observation()
            print(f"Length of Observations: {len(observations)}")
            timestep = TimeStep(observation=observations,
                                reward=rewards,
                                terminal=False)

        elif self.current_phase == "quest":
            #print("quest ok")
            #print(self.failed_quests, self.success_quests)
            #print(self.current_leader_idx, self.current_player_idx, player_name)
            #print(self.current_quest_votes, self.quest_sizes[self.current_quest_idx])
            #print(self.current_player_idx, self.current_players_on_quest)

            assert self.failed_quests < 3 and self.success_quests < 3
            assert len(self.current_players_on_quest) == self.quest_sizes[self.current_quest_idx]

            if len(self.current_quest_votes) == 0:
                #print(self.current_players_on_quest, self.current_quest_votes)
                self._moderator_speak(f"Current team proposal by Player {self.current_leader_idx}: {self.current_players_on_quest}")
                self._moderator_speak(f"This quest team has gotten majority votes. Now, all players on the quest can start this task.")

                quest_success_format = {'quest_idx': self.current_quest_idx, 'quest_player': self.current_player_idx, 'vote': 1}
                quest_failure_format = {'quest_idx': self.current_quest_idx, 'quest_player': self.current_player_idx, 'vote': 0}
                self._moderator_speak(f"For Players {self.current_players_on_quest}, you can vote for the quest now. You should not reveal your vote to other players. You only need to tell me your decision using following format: \n\nIf you vote for success: \n\n{json.dumps(quest_success_format)} \nOtherwise: \n\n{json.dumps(quest_failure_format)}. If your vote does not follow the format, your vote will be automatically discarded.")

            if len(self.current_quest_votes) < self.quest_sizes[self.current_quest_idx]:
                #print(self.anonymous_player_names[self.current_player_idx], self.current_players_on_quest)
                if self.anonymous_player_names[self.current_player_idx] in self.current_players_on_quest:
                    # We only need players on quest to generate messages
                    #pdb.set_trace()
                    message = Message(agent_name=player_name, content=action, turn=self.current_turn, visible_to=["Moderator"])
                    self.message_pool.append_message(message)

                    # Update votes based on player action
                    vote = self._text2vote(action)
                    self.current_quest_votes[self.anonymous_player_names[self.current_player_idx]] = vote
                    #print(self.current_quest_votes)
                else:
                    print(f"{self.current_player_idx} is not on quest. Skipping.")
                rewards = self._get_zero_rewards()
                
            else:
                print("all players have voted.")
                print(self.current_quest_votes)
                print(self.failed_quests, self.success_quests)
                if sum(self.current_quest_votes.values()) < self.quest_sizes[self.current_quest_idx]:
                    print("quest failed")
                    self.failed_quests += 1
                else:
                    print("quest succeeded")
                    self.success_quests += 1
                print(self.failed_quests, self.success_quests)
                
                # Check game status and update game phase
                if self.is_terminal():
                    if self.failed_quests >= 3:
                        rewards = self._get_rewards(good_win=False)
                    elif self.success_quests >= 3:
                        self.current_phase = "assassination"
                        rewards = self._get_zero_rewards()
                else:
                    self.current_phase = "team_building"
                    self.current_leader_idx = self._get_next_leader()
                    self.current_quest_idx = self._get_next_quest()
                    rewards = self._get_zero_rewards()
            
            self.current_turn += 1
            observations = self._get_observation()
            print(f"Length of Observations: {len(observations)}")
            timestep = TimeStep(observation=observations,
                                reward=rewards,
                                terminal=False)

        elif self.current_phase == "assassination":
            #print(self.current_leader_idx, self.current_player_idx, player_name)
            #assert self.failed_quests < 3 and self.success_quests >= 3
            #assert self.player_names[self.current_player_idx] == "Assasin"  # only assasin has the opportunity to perform an action
            
            # add transition sentences
            if self.success_quests < 3 or self.failed_quests >= 3:
                raise ValueError(f"Wrong phase! There should be at least three successful quests before assassination.")

            if self.player_names[self.current_player_idx] != "Assassin":
                rewards = self._get_zero_rewards()
                self.current_turn += 1
                observations = self._get_observation()
                print(f"Length of Observations: {len(observations)}")
                timestep = TimeStep(observation=observations,
                                    reward=rewards,
                                    terminal=False)
            else:
                assassination_format = {'player_idx': self.current_player_idx, 'player_to_assassinate': 'Player 1'}
                #assassination_format = "I choose to assassinate Player 1"
                self._moderator_speak(f"Assassin, now you have the last chance to assasinate Merlin. You should clearly announce which player you want to assassinate using this format: \n\n{assassination_format}. \n\nOtherwise your assassination will fail.")
                message = Message(agent_name=player_name, content=action, turn=self.current_turn,
                                  visible_to="all")
                self.message_pool.append_message(message)
                #print(message)

                # get assasinated_player
                self.assassinated_player = self._get_assassinated_player(action)
                #print(self.assassinated_player)
                #self.assassinated_player = "Player 0"
                #print(self.assassinated_player)

                #pdb.set_trace()
                if self.assassinated_player == f"Player {self._merlin_idx}":
                    self._moderator_speak(f"Merlin has been assasinated! Team Evil won!")
                    rewards = self._get_rewards(good_win=False)
                else:
                    self._moderator_speak(f"Merlin has survived from the assassination! Team Good won!")
                    rewards = self._get_rewards(good_win=True)

                print(rewards)
                observations = self._get_observation()
                print(f"Length of Observations: {len(observations)}")
                timestep = TimeStep(observation=observations,
                                    reward=rewards,
                                    terminal=True)
        else:
            raise ValueError(f"Unknown phase: {self.current_phase}")
        
        # Check if the player signals the end of the conversation
        if self.is_terminal():
            print("terminating!")
            self.message_pool.print()
            timestep.terminal = True

        return timestep

    @abstractmethod
    def check_action(self, action: str, player_name: str) -> bool:
        """
        Check whether a given action is valid for a player.

        Note:
            This method must be implemented by subclasses.

        Parameters:
            action (str): The action to be checked.
            player_name (str): The name of the player.

        Returns:
            bool: True if the action is valid, False otherwise.
        """
        ## TODO
        return True

    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Check whether the environment is in a terminal state (end of episode).

        Note:
            This method must be implemented by subclasses.

        Returns:
            bool: True if the environment is in a terminal state, False otherwise.
        """
        if self.success_quests >= 3 and self.assassinated_player is None: 
            self._terminal = False
        elif self.failed_quests >= 3 or self.assassinated_player is not None:
            self._terminal = True
        else:
            self._terminal = False
        return self._terminal

    def _get_zero_rewards(self) -> Dict[str, float]:
        """
        Return a dictionary with all player names as keys and zero as reward.

        Returns:
            Dict[str, float]: A dictionary of players and their rewards (all zero).
        """
        return {player: 0. for player in self.anonymous_player_names}

    def _get_rewards(self, good_win=None) -> Dict[str, float]:
        """
        Return a dictionary with all player names as keys and one as reward.

        Returns:
            Dict[str, float]: A dictionary of players and their rewards (all one).
        """
        if good_win == True:
            return {player: 1. if self.player_names[i] in TEAM_GOOD else 0. for i, player in enumerate(self.anonymous_player_names)}
        elif good_win == False:
            return {player: 1. if self.player_names[i] in TEAM_EVIL else 0. for i, player in enumerate(self.anonymous_player_names)}
