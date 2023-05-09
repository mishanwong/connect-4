import numpy as np
import random


class Connect4:
    ROW = 6
    COLUMN = 7

    def __init__(self):
        """
        Initialize game board.
        Each game board is a 6x7 board.
        """
        self.board = (
            (0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0),
        )
        self.player = 1  # There are 2 players, player 1 and player 2
        self.winner = None

    def available_actions(self, board):
        """
        This function takes a state of the board as input
        and return all the available actions (i, j) in that state.

        Action `(i, j)` represents dropping the coin in row `i` and column `j`.
        """
        actions = set()
        board = np.array(board)
        for j, col in enumerate(board.T):
            zero_indices = col.nonzero()[0]
            if zero_indices.size == 0:
                i = self.ROW - 1
            else:
                i = zero_indices[0] - 1
            actions.add((i, j))

        return actions

    def move(self, action):
        """
        Make a move `action` for the current player
        """
        # Update board
        row, col = action
        self.board[row][col] = self.player

        # Check for winner here
        if self.check_winner():
            self.winner = self.player

        # Check for tie here

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
            print(row)
            if self._check_arr(row):
                print("Row is true")
                return True

        # Check each column
        board = np.array(self.board)
        for i in range(len(board[0])):
            col = board[:, i]
            if self._check_arr(col):
                print("Col is true")
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
                print("Main diagonal is true")
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
                print("Secondary diagonal is true")
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

        return False


class Connect4AI:
    def __init__(self, alpha=0.5, epsilon=0.1):
        """
        Initialize AI with an empty Q-learning dictionary,
        an alpha (learning) rate, and an epsilon rate.

        The Q-learning dictionary maps `(state, action)`
        pairs to a Q-value (a number).
        - `state` is a tuple for (board, player)
        - `action` is a tuple `(i, j) that marks the position to be filled
        """
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

    def update(self, old_state, action, new_state, reward):
        """
        Update the Q-learning model.
        """
        old_q = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old_q, reward, best_future)

    def get_q_value(self, state, action):
        """
        Return the Q-value for the given `state` and `action`.
        If no Q-value exists yet in `self.q`, return 0.

        """
        if (state, action) not in self.q:
            return 0
        return self.q[(state, action)]

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """
        This function update the `self.q` dictionary.
        Use the formula:

        Q(s, a) <- old value estimate
                   + alpha * (new value estimate - old value estimate)

        where `old value estimate` is the previous Q-value,
        `alpha` is the learning rate, and `new value estimate`
        is the sum of the current reward and estimated future rewards.
        """
        new_q = old_q + self.alpha * (reward + future_rewards - old_q)
        key = (state, action)
        self.q[key] = new_q

    def best_future_reward(self, state):
        """
        Given a `state`, consider all possible `(state, action)`
        pairs available in that state and return the maximum of all of their
        Q-values.

        If a `(state, action)` pair has no Q-value in self.q, return 0.

        If there are no available actions in `state`, return 0.
        """
        q_value_list = []
        for key, value in self.q.items():
            s, _ = key
            if s == state:
                q_value_list.append(value)

        if not q_value_list:
            return 0

        return max(q_value_list)

    def choose_action(self, state, epsilon=True):
        """
        Given a `state`, return an action (i, j) to take.

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
            key = (state, action)
            if key in self.q:
                action_value_pairs.append((action, self.q[key]))
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
        return action_value_pairs[0][0]


def train(n):
    """
    Tran an AI by playing `n` games against itself.
    """
    ai = Connect4AI()

    for i in range(n):
        print(f"Playing training game {i + 1}")
        game = Connect4()

        # Keep track of last move made by either player
        last = {
            # Player 1
            1: {"state": None, "action": None},
            # Player 2
            2: {"state": None, "action": None},
        }

        # Game loop
        while True:
            # Keep track of current state and action
            state = game.board.copy()
            action = ai.choose_action(game.board)

            # Keep track of last state and action
            last[game.player]["state"] = state
            last[game.player]["action"] = action

            # Make a move
            game.move(action)
            new_state = game.board.copy()

            # If there is a winner, update Q values with rewards
            if game.winner is not None:
                ai.update(state, action, new_state, 1)
                other_player = game.get_other_player()
                other_player_last_state = last[other_player]["state"]
                other_player_last_action = last[other_player]["action"]
                ai.update(
                    other_player_last_state, other_player_last_action, new_state, -1
                )
                break

            # If game is continuing, no rewards yet
            elif True:
                pass  # TODO

    print("Done training")

    # Return the trained AI
    return ai


game = Connect4()
b = (
    (0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 1),
    (0, 1, 2, 1, 0, 1, 0),
)

game_over = (
    (1, 1, 1, 1, 1, 1, 1),
    (1, 1, 1, 1, 1, 1, 1),
    (1, 1, 1, 1, 1, 1, 1),
    (1, 1, 1, 1, 1, 1, 1),
    (1, 1, 1, 1, 1, 1, 1),
    (1, 1, 1, 1, 1, 1, 0),
)

hello = game.available_actions(b)
print(hello)
