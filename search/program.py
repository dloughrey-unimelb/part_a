# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part A: Single Player Cascade

import heapq
from collections import deque
from .core import CellState, Coord, Direction, Action, MoveAction, EatAction, CascadeAction, PlayerColor, BOARD_N
from .utils import render_board

BoardState = frozenset[tuple[Coord, CellState]]

def board_dict_to_state(board: dict[Coord, CellState]) -> BoardState:
    return frozenset(board.items())

def state_to_dict(state: BoardState) -> dict[Coord, CellState]:
    return dict(state)

def heuristic(state: BoardState) -> int:
    return sum(
        cell.height
        for _, cell in state
        if cell.color == PlayerColor.BLUE
    )


def search(
    board: dict[Coord, CellState]
) -> list[Action] | None:
    """
    This is the entry point for your submission. You should modify this
    function to solve the search problem discussed in the Part A specification.
    See `core.py` for information on the types being used here.

    Parameters:
        `board`: a dictionary representing the initial board state, mapping
            coordinates to `CellState` instances (each with a `.color` and
            `.height` attribute).

    Returns:
        A list of actions (MoveAction, EatAction, or CascadeAction), or `None`
        if no solution is possible.
    """

    # The render_board() function is handy for debugging. It will print out a
    # board state in a human-readable format. If your terminal supports ANSI
    # codes, set the `ansi` flag to True to print a colour-coded version!
    print(render_board(board, ansi=True))

    # Do some impressive AI stuff here to find the solution...
    # ...
    # ... (your solution goes here!)
    # ...

    return bfs_search(board)

    

def a_star_search(
    board: dict[Coord, CellState]
) -> list[Action] | None:
    
    initial_state = board_dict_to_state(board)

    # Goal: no Blue tokens remain
    if heuristic(initial_state) == 0:
        return []
    
    counter = 0

    start_h = heuristic(initial_state)

    open_set: list[tuple[int, int, int, BoardState, list[Action]]] = [
        (start_h, 0, counter, initial_state, [])
    ]

    closed_set: set[BoardState] = set()


    while open_set :
        f, g, _, state, actions = heapq.heappop(open_set)

        if state in closed_set :
            continue

        closed_set.add(state)

        for action, next_state in get_neighbors(state) :

            if next_state in closed_set :
                continue

            new_g = g + 1
            new_h = heuristic(next_state)
            new_f = new_g + new_h
            new_actions = actions + [action] 

            if new_h == 0 :
                return new_actions

            counter += 1
            heapq.heappush(open_set, (new_f, new_g, counter, next_state, new_actions))

    return None


def bfs_search(
    board: dict[Coord, CellState]
) -> list[Action] | None:
    
    initial_state = board_dict_to_state(board)
    queue = deque()
    queue.append((initial_state, []))

    while queue :
        state, actions = queue.popleft()

        if is_goal(state):
            return actions

        for action, new_state in get_neighbors(state):
            queue.append((new_state, actions + [action]))
    
    return None

def is_goal(state: BoardState) -> bool:
    return not any(cell.color == PlayerColor.BLUE for coord, cell in state)

def get_neighbors(state: BoardState) -> list[tuple[Action, BoardState]]:
    
    board = state_to_dict(state)
    results = []

    for coord, cell in board.items():
        if cell.color != PlayerColor.RED:
            continue

        for direction in Direction:
            # MOVE
            new_board = apply_move(board, coord, direction)
            if new_board is not None:
                results.append((
                    MoveAction(coord, direction),
                    board_dict_to_state(new_board)
                ))

            # EAT
            new_board = apply_eat(board, coord, direction)
            if new_board is not None:
                results.append((
                    EatAction(coord, direction),
                    board_dict_to_state(new_board)
                ))

            # CASCADE (only for height >= 2)
            if cell.height >= 2:
                new_board = apply_cascade(board, coord, direction)
                if new_board is not None:
                    results.append((
                        CascadeAction(coord, direction),
                        board_dict_to_state(new_board)
                    ))

    return results
    



def apply_move(board: dict[Coord, CellState], coord: Coord, direction: Direction) -> list[tuple[Action, dict[Coord, CellState]]]:

    try:
        dest = coord + direction   # raises ValueError if off the board
    except ValueError:
        return None

    src_cell = board.get(coord)
    if src_cell is None or src_cell.is_empty:
        return None

    dest_cell = board.get(dest, CellState())  # empty CellState if nothing there

    # Cannot MOVE onto an enemy stack
    if dest_cell.is_stack and dest_cell.color == PlayerColor.BLUE:
        return None

    new_board = dict(board)
    new_board.pop(coord, None)

    if dest_cell.is_empty:
        # Relocate
        new_board[dest] = src_cell
    else:
        # Merge with friendly
        new_board[dest] = CellState(PlayerColor.RED, src_cell.height + dest_cell.height)

    return new_board

def apply_eat(board: dict[Coord, CellState], coord: Coord, direction: Direction) -> list[tuple[Action, dict[Coord, CellState]]]:

    try:
        dest = coord + direction
    except ValueError:
        return None

    src_cell = board.get(coord)
    if src_cell is None or src_cell.is_empty:
        return None

    dest_cell = board.get(dest, CellState())

    # Must eat a blue stack, and attacker must be tall enough
    if not dest_cell.is_stack or dest_cell.color != PlayerColor.BLUE:
        return None
    if src_cell.height < dest_cell.height:
        return None

    new_board = dict(board)
    new_board.pop(coord, None)
    # Attacker moves to dest
    new_board[dest] = src_cell

    return new_board

def apply_cascade(board: dict[Coord, CellState], coord: Coord, direction: Direction) -> list[tuple[Action, dict[Coord, CellState]]]:

    src_cell = board.get(coord)
    if src_cell is None or src_cell.is_empty:
        return None
    if src_cell.height < 2:
        return None  # Cannot cascade a stack of height 1
    
    h = src_cell.height
    new_board = dict(board)
    new_board.pop(coord, None)

    for i in range(1, h + 1):
        # Compute target cell for this token
        dr = direction.r * i
        dc = direction.c * i
        tr, tc = coord.r + dr, coord.c + dc

        # Token falls off the board — eliminated
        if not (0 <= tr < BOARD_N and 0 <= tc < BOARD_N):
            continue

        target = Coord(tr, tc)

        # Push any existing stack at target further along the direction
        push_stack(new_board, target, direction)

        # Place one RED token at target
        existing = new_board.get(target, CellState())
        if existing.is_empty:
            new_board[target] = CellState(PlayerColor.RED, 1)
        else:
            # Should not happen after pushing, but merge defensively
            new_board[target] = CellState(existing.color, existing.height + 1)

    return new_board
    
def push_stack(board: dict[Coord, CellState], coord: Coord, direction: Direction) -> None:
    
    cell = board.get(coord, CellState())
    if cell.is_empty:
        return  # Nothing to push

    dr, dc = direction.r, direction.c
    nr, nc = coord.r + dr, coord.c + dc

    # Push off the board — eliminate the stack
    if not (0 <= nr < BOARD_N and 0 <= nc < BOARD_N):
        del board[coord]
        return

    next_coord = Coord(nr, nc)

    # Recursively push whatever is at the next cell first
    push_stack(board, next_coord, direction)

    # Move this stack to next_coord
    board[next_coord] = cell
    del board[coord]
