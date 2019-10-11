import numpy
board = numpy.zeros((6,7))
firstMove = 1;
secondMove = -1;
def moveCount():
        count = 0
        for c in range(7):
                for r in range(6):
                        if (board[r][c] != 0):
                                count += 1
        return count
                                
def hasWon(player):
        #Check horizontal locations
        for c in range(4):
                for r in range(6):
                        if board[r][c] == board[r][c+1] == board[r][c+2] == board[r][c+3] == player:
                                return player

        #Check vertical locations
        for c in range(7):
                for r in range(3):
                        if board[r][c] == board[r+1][c] == board[r+2][c] == board[r+3][c] == player:
                                return player

        #Check positively sloped diaganol locations
        for c in range(4):
                for r in range(3):
                        if board[r][c] == board[r+1][c+1] == board[r+2][c+2] == board[r+3][c+3] == player:
                                return player

        #Check negatively sloped diaganol locations
        for c in range(4):
                for r in range(3, 6):
                        if board[r][c] == board[r-1][c+1] == board[r-2][c+2] == board[r-3][c+3] == player:
                                return player
        if (moveCount() == 42):
                return 0.5
        else:
                return 0
def reward(player):
        re = 0
        for c in range(5):
                for r in range(6):
                        if board[r][c] == board[r][c+1] == board[r][c+2] == player:
                                re += 1

        #Check vertical locations
        for c in range(7):
                for r in range(4):
                        if board[r][c] == board[r+1][c] == board[r+2][c] == player:
                                re += 1

        #Check positively sloped diaganol locations
        for c in range(5):
                for r in range(4):
                        if board[r][c] == board[r+1][c+1] == board[r+2][c+2] == player:
                                re += 1

        #Check negatively sloped diaganol locations
        for c in range(5):
                for r in range(2, 6):
                        if board[r][c] == board[r-1][c+1] == board[r-2][c+2] == player:
                                re += 1
        for c in range(6):
                for r in range(6):
                        if board[r][c] == board[r][c+1] == player:
                                re += 4

        #Check vertical locations
        for c in range(7):
                for r in range(5):
                        if board[r][c] == board[r+1][c] == player:
                                re += 4

        #Check positively sloped diaganol locations
        for c in range(6):
                for r in range(5):
                        if board[r][c] == board[r+1][c+1] == player:
                                re += 4

        #Check negatively sloped diaganol locations
        for c in range(6):
                for r in range(1, 6):
                        if board[r][c] == board[r-1][c+1] == player:
                                re += 4
        return re
                        
def dropTile(columnNum, player):
        val = -1
        for rowNum in range(5, -1, -1):
                if (board[rowNum][columnNum] == 0):
                        board[rowNum][columnNum] = player
                        val = 1
                        break;
        return val
def stateSize():
        return 42
def actionSize():
        return 7
def reset():
        global board
        board = numpy.zeros((6,7))
def render():
        print(board)
def set(boardy):
        global board
        board = boardy
