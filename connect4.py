from typing import List
import numpy as np
import random
import time
import copy


class Connect4:
    ROW = 6
    COLUMN = 7

    def __init__(self):
        """
        Initialize game board.
        Each game board is a 6x7 board.
        """
        self.board = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
        self.player = 1  # There are 2 players, player 1 and player 2
        self.winner = None

    def available_actions(self, board):
        """
        This function takes a state of the board as input
        and return all the available `actions` in that state.

        `action` is an integer that represents dropping the coin in column `action`.
        """
        actions = set()
        board = np.array(board)
        for j, col in enumerate(board.T):
            i = self.find_lowest_zero_index(col)
            if i >= 0:
                actions.add(j)

        return actions

    def move(self, action: int):
        """
        Make a move by dropping a coin into the column `action`
        for the current player
        """
        # Update board
        column = np.array(self.board)[:, action]
        row = self.find_lowest_zero_index(column)
        self.board[row][action] = self.player

        # Check for winner here
        if self.check_winner():
            self.winner = self.player

        # Switch player
        self.switch_player()

    def get_other_player(self):
        if self.player == 1:
            return 2
        else:
            return 1

    def switch_player(self):
        """
        Switch the current player to the other player.
        There are 2 players - Player 1 and player 2
        """
        self.player = self.get_other_player()

    def check_game_over(self):
        """
        Check if game if over (ie. all cells are filled)
        """
        # Check if all cells are filled
        arr = np.array(self.board)

        game_is_over = np.all(arr != 0)

        return game_is_over

    def check_winner(self):
        """
        Check if there is a winner and return the winner if there is, otherwise return None
        """
        # Check each row
        for row in self.board:
            if self._check_arr(row):
                return True

        # Check each column
        board = np.array(self.board)
        for i in range(len(board[0])):
            col = board[:, i]
            if self._check_arr(col):
                return True

        # Check each diagonal
        # 1. Main diagonal
        # Going down the row
        main_diagonal = []
        starting_positions_main = set()

        # Fill in starting positions rows
        for row in range(self.ROW):
            starting_positions_main.add((row, 0))

        # Fill in starting positions cols
        for col in range(self.COLUMN):
            starting_positions_main.add((0, col))

        # Create arrays of the main diagonal
        for pos in starting_positions_main:
            i, j = pos
            positions = []
            values = []
            while i < self.ROW and j < self.COLUMN:
                positions.append((i, j))
                values.append(self.board[i][j])
                i += 1
                j += 1

            main_diagonal.append(values)

        # Check each main diagonal
        for diag in main_diagonal:
            if self._check_arr(diag):
                return True

        # Create arrays for secondary diagonal
        secondary_diagonal = []
        starting_positions_secondary = set()

        for row in range(self.ROW):
            starting_positions_secondary.add((row, self.COLUMN - 1))

        for col in range(self.COLUMN):
            starting_positions_secondary.add((0, col))

        # Create arrays of the secondary diagonal
        for pos in starting_positions_secondary:
            i, j = pos
            positions = []
            values = []

            while i < self.ROW and j >= 0:
                positions.append((i, j))
                values.append(self.board[i][j])
                i += 1
                j -= 1
            secondary_diagonal.append(values)

        # Check each secondary diagonal
        for diag in secondary_diagonal:
            if self._check_arr(diag):
                return True

        # return False in the end
        return False

    def _check_arr(self, arr):
        """
        Helper function for check_winner function
        Check whether there 4 consecutive same kind in a row
        """
        current = None
        count = 1

        for item in arr:
            if item != 0:
                if item == current:
                    count += 1
                    if count >= 4:
                        return True
                else:
                    current = item
                    count = 1
            else:
                count = 0

        return False

    def find_lowest_zero_index(self, column):
        zero_indices = column.nonzero()[0]
        if zero_indices.size == 0:
            zero_idx = self.ROW - 1
        else:
            zero_idx = zero_indices[0] - 1
        return zero_idx


class Connect4AI:
    def __init__(self, epsilon=0.1):
        """
        Initialize AI with an empty Q-learning dictionary,
        number of training runs, and an epsilon rate.

        TODO: Explain dictionary q structure
        """
        self.q = dict()
        self.epsilon = epsilon

    def update(
        self,
        old_state: List[List[int]],
        action: int,
        new_state: List[List[int]],
        reward: int,
        alpha: float,
    ):
        """
        Update the Q-learning model.
        """
        old_q: float = self.get_q_value(old_state, action)
        best_future: float = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old_q, reward, best_future, alpha)

    def get_q_value(self, state: List[List[int]], action: int):
        """
        Return the Q-value for the given `state`, `player` and `action`.
        If no Q-value exists yet in `self.q`, return 0.

        """
        state_key = self.list_to_tuple(state)

        if state_key not in self.q:
            return 0

        if state_key in self.q and action not in self.q[state_key]:
            return 0

        return self.q[state_key][action]

    def update_q_value(
        self,
        state: List[List[int]],
        action: int,
        old_q: float,
        reward: int,
        future_rewards: float,
        alpha: float,
    ):
        """
        This function update the `self.q` dictionary.
        Use the formula:

        Q(s, a) <- old value estimate
                   + alpha * (new value estimate - old value estimate)

        where `old value estimate` is the previous Q-value,
        `alpha` is the learning rate, and `new value estimate`
        is the sum of the current reward and estimated future rewards.
        """
        new_q = old_q + alpha * (reward + future_rewards - old_q)
        state_key = self.list_to_tuple(state)
        if state_key in self.q:
            self.q[state_key][action] = new_q
        else:
            self.q[state_key] = {action: new_q}

    def best_future_reward(self, state):
        """
        Given a `state`, consider all possible `action` for that state
        and return the maximum of all of their
        Q-values.

        If a `state` and `action` combination has no Q-value in self.q, return 0.

        If there are no available actions in `state`, return 0.
        """
        q_value_list = []
        # pretty_print_nonzero_q(self.q)
        for state_key, actions in self.q.items():
            # print("actions", actions)
            if state_key == self.list_to_tuple(state):
                for action, q_val in actions.items():
                    q_value_list.append(q_val)

        if not q_value_list:
            return 0

        return max(q_value_list)

    def choose_action(self, state, epsilon=True):
        """
        Given a `state`, return an `action` that indicates which column
        to drop the coin into.

        If `epsilon` is `False`, then return the best action
        available in the state (the one with the highest Q-value,
        using 0 for pairs that have no Q-values) - This is the
        greedy approach.

        If `epsilon` is True, then with probability `self.epsilon`
        choose a random available action,
        otherwise choose the best action.

        If multiple actions have the same Q-value, choose any of
        those options.
        """
        connect4 = Connect4()

        action_value_pairs = []
        available_actions = connect4.available_actions(state)

        for action in available_actions:
            state_key = self.list_to_tuple(state)
            if state_key in self.q:
                if action in self.q[state_key]:
                    action_value_pairs.append((action, self.q[state_key][action]))
                else:
                    action_value_pairs.append((action, 0))
            else:
                action_value_pairs.append((action, 0))

        # Sort the action value pairs in descending order by q-value
        action_value_pairs = sorted(
            action_value_pairs, key=lambda x: x[1], reverse=True
        )

        if epsilon:
            # Add some randomness with probability epsilon
            rand_num = random.random()
            if rand_num < self.epsilon:
                return random.choice(action_value_pairs)[0]

        # Return the action with the largest q-value.
        # If multiple actions have the same q-value, return any or them randomly.
        action_list = []
        max_q_val = float("-inf")

        for _action, q_val in action_value_pairs:
            if q_val > max_q_val:
                action_list = [_action]
                max_q_val = q_val
            elif q_val == max_q_val:
                action_list.append(_action)

        return random.choice(action_list)

    def temp_sorted_actions_list(self, state, epsilon=True):
        """Function for testing purpose only
        to be removed
        """
        connect4 = Connect4()

        action_value_pairs = []
        available_actions = connect4.available_actions(state)

        for action in available_actions:
            state_key = self.list_to_tuple(state)
            if state_key in self.q:
                if action in self.q[state_key]:
                    action_value_pairs.append((action, self.q[state_key][action]))
                else:
                    action_value_pairs.append((action, 0))
            else:
                action_value_pairs.append((action, 0))

        # Sort the action value pairs in descending order by q-value
        action_value_pairs = sorted(
            action_value_pairs, key=lambda x: x[1], reverse=True
        )
        return action_value_pairs

    def list_to_tuple(self, list):
        """
        Given a state of the board as a 2-D array, convert the state
        to a 2D tuple so that it can be used as a dictionary key.
        """
        return tuple(tuple(row) for row in list)


def train(n):
    """
    Train an AI by playing `n` games against itself.
    """
    ai = Connect4AI()

    for i in range(n):
        if (i + 1) % 100 == 0:
            print(f"### Playing training game {i + 1} ###")

        game = Connect4()

        # Keep track of last move made by either player
        last = {
            # Player 1
            1: {"state": None, "action": None},
            # Player 2
            2: {"state": None, "action": None},
        }

        # Calculate alpha - reduce proportionately to number of training runs
        alpha = 0.1 - (0.1 / n) * i

        # Game loop
        while True:
            # Keep track of current state and action
            state = copy.deepcopy(game.board)
            action: int = ai.choose_action(game.board)

            # Keep track of last state and action
            last[game.player]["state"] = state
            last[game.player]["action"] = action

            # Make a move
            game.move(action)

            new_state = copy.deepcopy(game.board)

            # If there is a winner, update Q values with rewards
            if game.winner is not None:
                # Update Q-value for the winning move
                ai.update(state, action, new_state, 1, alpha)

                # Update Q-value for the losing move
                losing_state = last[game.player]["state"]
                losing_action = last[game.player]["action"]
                ai.update(losing_state, losing_action, state, -1, alpha)
                break

            # If there the board is filled and no winner, then no reward
            elif game.check_game_over() and game.winner is None:
                break

            # If game is continuing, no rewards yet
            else:
                ai.update(state, action, new_state, 0, alpha)

    print(f"Done training")

    # Return the trained AI
    return ai


def play(ai, human_player=None):
    """
    Play a human game against the AI
    `human_player` are assigned as player 1 or player 2 randomly
    """
    if human_player is None:
        human_player = random.randint(1, 2)

    # Create a new game
    game = Connect4()

    for key, value in ai.q.items():
        print_board(key)
        for action, q_val in value.items():
            print(action, q_val)

    # Game loop
    while True:
        # Compute available actions
        available_actions = game.available_actions(game.board)
        # print(available_actions)
        time.sleep(1)

        print("sorted action", ai.temp_sorted_actions_list(game.board))

        # Let human make a move
        if game.player == human_player:
            print()
            print(f"Your Turn. You are player {human_player}.")
            print_board(game.board)
            while True:
                col_num = int(input("Choose Column: "))
                if col_num in available_actions:
                    break
                print("Invalid move, try again.")

        # Have AI make a move
        else:
            print()
            ai_player = 2 if human_player == 1 else 1
            print(f"AI's Turn. AI is player {ai_player}.")
            col_num = ai.choose_action(game.board)
            print(f"AI chose to drop coin in column {col_num}")

        # Make a move
        game.move(col_num)

        # Print the board
        print_board(game.board)

        # Check for winner
        if game.winner is not None:
            print()
            print("GAME OVER")
            winner = "Human" if game.winner == human_player else "AI"
            print(f"Winner is {winner}")
            return


game = Connect4()


def print_board(board):
    print("COLUMN: ", [i for i in range(game.COLUMN)])
    for idx, row in enumerate(board):
        print(f"     {idx}  ", row)


def pretty_print_nonzero_q(q):
    for state_key, actions in q.items():
        for action, q_val in actions.items():
            if q_val != 0:
                print_board(state_key)
                print("Action: ", action)
                print("Q-value", q_val)
                print()
