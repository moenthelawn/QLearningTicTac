'''
Created on Aug 3, 2018

@author: Chris
'''
import numpy as np 
from _operator import length_hint
import random 
import copy 
from ctypes import c_int64
class State: 
    
    def __init__(self,stateNumber,moves,rewards,board):
        self.stateNumber = stateNumber
        self.moves = [[1,1,1],[1,1,1],[1,1,1]]
        self.rewards = rewards 
        self.board = board 
        self.Q_max = 0 
        self.i = -1 #Max row action 
        self.j = -1  #Max column action 
        
class StateNumber:
    def __init__(self): 
        self.stateNumberX = 0 
        self.stateNumberO = 0     
def checkLength(board,characterType, i,j):
    #Check row across 
    l =3
    counter_row_up = 0 
    counter_row_down = 0 
    
    rows = len(board) - 1
    cols = len(board[0]) - 1
    
    
    counter_diagonal_top_right = 0 
    counter_diagonal_top_left = 0 
    counter_diagonal_bottom_right = 0
    counter_diagonal_bottom_left = 0 
    
    counter_col_right = 0 
    counter_col_left = 0 
    
    drawCounter = 0
    
    
    for k in range(l): 
        
        if (i + k) <= rows: 
            if board[i + k][j] == characterType: 
                counter_row_up = counter_row_up + 1 
                
            if (j + k) <= cols: 
                if board[i + k][j + k] == characterType: 
                      #Diagonal down right 
                    counter_diagonal_bottom_right = counter_diagonal_bottom_right + 1
           
                
            if (j - k) >= 0:
                if board[i + k][j - k] == characterType: 
                    #Downwards 
                    counter_diagonal_bottom_left = counter_diagonal_bottom_left + 1
                    
        if (i - k) >= 0: 
            if board[i - k][j] == characterType: 
                counter_row_down = counter_row_down + 1
           
                
            if (j + k) <= cols: 
                if board[i - k][j + k] == characterType:
                    #diagonal top right
                    counter_diagonal_top_right = counter_diagonal_top_right + 1
       
            if (j - k) >=0: 
                if board[i - k][j - k] == characterType: 
                    # diagonal up left 
                    counter_diagonal_top_left = counter_diagonal_top_left + 1
                    
        if (j + k) <= cols: 
            if board[i][j+k] ==characterType: 
                counter_col_right=  counter_col_right+1
                
        if (j - k) >= 0: 
            if board[i][j-k] == characterType: 
                counter_col_left = counter_col_left + 1
      
      
        # Upwards
        
       #Diagonal bottom left 
        #if board[i + k][j - k] == characterType and (i + k) <= 5 and (j - k) >= 0: 
          #  counter_diagonal_bottom_left = counter_diagonal_bottom_left + 1
    if  counter_row_up == l or counter_row_down == l or counter_diagonal_top_right == l or counter_diagonal_top_left == l or counter_diagonal_bottom_right == l or  counter_diagonal_bottom_left == l or counter_col_right == l or counter_col_left == l:
        #If there is a sequence such that the game has be won      
        return 1 
    elif drawCounter == 8: 
        return 2 #This represents a draw
    else: 
        #The game has not yet been won 
        return 0  

def checkWinner(board, x , y ):
    
    #check if previous move caused a win on vertical line 
    if board[0][y] == board[1][y] == board [2][y]:
        return True

    #check if previous move caused a win on horizontal line 
    if board[x][0] == board[x][1] == board [x][2]:
        return True

    #check if previous move was on the main diagonal and caused a win
    if x == y and board[0][0] == board[1][1] == board [2][2]:
        return True

    #check if previous move was on the secondary diagonal and caused a win
    if x + y == 2 and board[0][2] == board[1][1] == board [2][0]:
        return True

    return False        
    
def isInList(value,list): 
    rows = len(list) 
    cols = len(list[0])
    
    for i in range(rows): 
        for j in range(cols):
            if value == list[i][j]: 
                return True 
            
    return False 

def resetBoard(board):#Simple function to reset the board to its original state
    rows = len(board) 
    cols = len(board[0])
    for i in range(rows): 
        for j in range(cols): 
            board[i][j] = '*'
    return board 
    
#Game not won yet = 0 
# Game won by Xs = 1 
# Game won by Os = 2 
# Game Draw = 3
def calculateState(i1,j1,board,statePlayerX,statePlayerO,stateNumber,player): 
  

    #First we check the Xs 
    if (checkWinner(board,i1,j1) == True): 
        #Check the ones around it 
        print("The game has been won by player ",player,"!")
        #We want to update the reward of the current state of the computer 
        #if player == 'X':
          #  states[stateNumber].rewards[i][j] += 1 
       # elif player == 'O': 
       #     states[stateNumber].rewards[i][j] -= 1
       #If that was the player who won 
        if board[i1][j1] == 'X': 
            statePlayerX[stateNumber[0]].rewards[i1][j1] += 1 
            statePlayerO[stateNumber[0]].rewards[i1][j1] -= 1 
        elif board[i1][j1] == 'O': 
            statePlayerX[stateNumber[0]].rewards[i1][j1] -= 1 
            statePlayerO[stateNumber[0]].rewards[i1][j1] += 1 
        stateNumber[0] += 1 
        board = resetBoard(board)
    elif (isInList('*', board) == False): 
        #If the board is such that no further moves can be made, then we will declare the draw and add the reward of 0.5 to each of the player 
        statePlayerO[stateNumber[0]].rewards[i1][j1] += 0.5 
        statePlayerX[stateNumber[0]].rewards[i1][j1] += 0.5 
        board = resetBoard(board)
        stateNumber[0] += 1 
 

board = [ ['*','*','*'],
         ['*','*','*'],
         ['*','*','*'],
    ]
testXWin = [ ['*','O','O'],
             ['*','X','*'],
             ['*','*','X']]


state = {} #State action 

def allowableActions(board):
    #This function loops through and finds all of the allowable actions based on the current board 
    board_allowable_actions = [
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],]
    rows = len(board) 
    cols = len(board[0])
    for i in range(rows): 
        for j in range(cols): 
            if board[i][j] == '*':
                board_allowable_actions[i][j] = 1
    return board_allowable_actions
  
def checkAllStates(states,board):
    length = len(states)
    for i in range(length): 
        if board == states[i].board:
            return i  
    return -1 
def biwiseAndFunc(validMoves,rewards):
    rows = len(validMoves)
    cols = len(validMoves[0])
    Matrix = [[0 for x in range(rows)] for y in range(cols)]
    for i in range(rows): 
        for j in range(cols):
            validMove = validMoves[i][j]
            Reward = rewards[i][j]
           # Matrix[i][j] = validMoves[i][j] * rewards[i][j]
            Matrix[i][j] = validMove * Reward 
    return Matrix 

def getMaxAction(rewards,validMoves):
    maxReward = 0
    rows = len(rewards)
    cols = len(rewards[0]) 
    maxI= 0 
    maxJ = 0 
    for i in range(rows): 
        for j in range(cols): 
            currentReward = rewards[i][j] 
            if currentReward > maxReward and validMoves[i][j] != 0: 
                maxReward = currentReward
                maxI = i 
                maxJ = j 
    return [maxI,maxJ]
    
    
def chooseMove(validMoves, rewards):
    epsilon = 0.9
    randNumber = random.uniform(0, 1)
    
    okayMove  = True 
    
    if randNumber > epsilon: 
        #Then we will choose a random move 
        
        #Although we must first make sure that the move is a valid move
        while(okayMove): 
            [i,j] = [random.randint(0,2),random.randint(0,2)]
           # print("R ", i, " " ,j)
            if (rewards[i][j] * validMoves[i][j]) != 0: 
                #If the valid move matrix at the index i,j is not zero (in other words 0 * rewards[i][j] is not zero), then this move 
                # is deemed valid 
                okayMove = False 
                #We need to update the valuid move matrix x
                return [i,j]
            
    else: 
        #Bitwise and the rewards and the valid moves such that we are left with the remaining moves and their rewards 
        #potentialRewardMoves = biwiseAndFunc(validMoves,rewards)
        #Now we need to get the max action of the 
        #maxAction_currentState = getMaxAction(valide)
        return getMaxAction(validMoves,rewards)

def getQPrimed(board, states ):
    stateExists = checkAllStates(states, board)
    if stateExists == -1: 
        #Then we do not have that state and thus the Q value for that move will be zero 
        return 0 
    else: 
        #Then we will find the Q value for that state 
        return states[stateExists].Q_max #Return the Q value for that state 
def updateQ(states,stateNumber,Q_primed):
    discountFactor = 0.9 
    lamb = 0.5 
       
    i = states[stateNumber].i 
    j = states[stateNumber].j 
    
    
    maxReward = states[stateNumber].rewards[i][j]
    states[stateNumber].Q_max = states[stateNumber].Q_max +  lamb * (maxReward + (discountFactor * Q_primed) - states[stateNumber].Q_max)
def displayBoard(rowI,rowJ,player, board):
    rows = len(board) 
    cols = len(board[0])
    
    print("Computer ",player," makes the move on row ", rowI, " and column ",rowJ)
    
  #  for i in range(numrows): 
   #     for j in range(numcols): 
    #        print(board[j][i],' ') 
            
   #     print("\n")
        
    print('\n'.join([''.join(['{:4}'.format(item) for item in row]) 
      for row in board]))
    print("--------------------------------------------------")
        
                 
def QlearnUpdate(states,stateNumber,board,player,list_playerX,list_playerO):
     
    #State exist will return either -1 or the correct index for the associated board 
    stateExists = [checkAllStates(states,board)]
    if stateExists[0] == -1: 
        #if our state does not exist, then we append the current state number with the current one 
        
        states[stateNumber[0]].state = stateNumber[0]
     
        potentialMoves = allowableActions(board)
        
        newBoard = copy.deepcopy(board)
        
        states[stateNumber[0]].board = newBoard
        states[stateNumber[0]].moves = potentialMoves #Making sure to update the current state with the appropriate moves we would be able to take 
        
        [i,j] = chooseMove(potentialMoves, states[stateNumber[0]].rewards) #Choose the actio i and j that would maximize the reward of the board 
        states[stateNumber[0]].i = i 
        states[stateNumber[0]].j = j 

        board[i][j] = player #Update the board position of the new targeted move 
        displayBoard(i,j,player,board)
       
        
        Q_primed = getQPrimed(board,states)
        
        updateQ(states, stateNumber[0], Q_primed) #Now We need to update the function of the current state 
        calculateState(i, j, board, list_playerX,list_playerO, stateNumber,player) #Check to see if O has won 
        stateNumber[0] += 1#Now we inrement the state number by 1 
      
    else: 
       
        #print(state[stateExists].moves, states[stateExists].rewards)
        #allowableMoves = allowableActions(currentBoard)
       #allowableMoves = state[stateExists].moves

        [i,j] = chooseMove(allowableActions(board), states[stateExists[0]].rewards)#Choose valid action
        states[stateExists[0]].i = i 
        states[stateExists[0]].j = j
        
        board[i][j] = player #Update the board of the max position with the new player's character 
        displayBoard(i,j,player,board)
        
        
        Q_primed = getQPrimed(board,states)
        updateQ(states, stateExists[0], Q_primed) #Now We need to update the function of the current state 
        
        calculateState(i, j, board, list_playerX,list_playerO, stateExists,player) #Check to see if O has won 
       
    
    
    #Choose the action from S using the policy derived from Q
    
    #Repeat for each step of the episode 
    #We take in all of hte actions of S' and observe the best possible action 
    
    #validMoves = S_prime.moves 
def randomRewardFunction(rewards):
    rows = 3 
    cols = 3 
    maxR = 10 
    
    for i in range(rows): 
        for j in range(cols): 
            rewards[i][j] = random.uniform(-1 * maxR,maxR)
    return rewards
def initializeList():
    
    length = 19683  
    list = []
    

    emptyBoard =  [ ['*','*','*'],
                   ['*','*','*'],
                   ['*','*','*'],
                   ]
    for i in range(length): 
    # list.insert(i, state(i,moves,rewards))\
        rewards = [[0,0,0],[0,0,0],[0,0,0]]
        moves = [[1,1,1],[1,1,1],[1,1,1]]
        rewards = randomRewardFunction(rewards)
        list.insert(i, State(i,moves,rewards,emptyBoard))
    
    return list 
def makeComputerMove(computerLUT,board):
    stateExists = checkAllStates(computerLUT,board) #Look up to see if we have that state 
    #We return i and j of the associated state 
    i_comp = computerLUT[stateExists].i 
    j_comp = computerLUT[stateExists].j
    return [i_comp,j_comp]
        #Then we know his state does not exist
def calculateBoardWin(board,i,j,player):
    #This function will calculate whether or not a game has been won 
    
    #First we check the Xs 
    if (checkWinner(board,i,j) == True): 
        #Check the ones around it 
        print("The game has been won by player ",player,"!")
        board = resetBoard(board)
    elif (isInList('*', board) == False): 
        board = resetBoard(board)
        
 
    
        
    
    
    
      
gameLoop = True 



stateNumber_X = [0] 
stateNumber_O = [0]

list_playerX = initializeList() #Initialize the list of the first player 
list_playerO = initializeList() #Initialie the list of the second player 

totalReward = 0 

while(gameLoop): 
    
    QlearnUpdate(list_playerX,stateNumber_X,board,'X',list_playerX,list_playerO) #Q learn update for the first AI player 
   # (states,stateNumber,board,player,list_playerX,list_playerO)
    #stateNumber += 1
    QlearnUpdate(list_playerO,stateNumber_O,board,'O',list_playerX,list_playerO) #Q learn update for for the second AI player 
    #calculateState(list_playerO[state_playerO].i, list_playerX[state_playerO].j, board, list_playerX,list_playerO, state_playerO,"O",stateNumber) #Check to see if O has won 
    print(stateNumber_O[0],stateNumber_X[0])
    if stateNumber_X[0] == 19683:
        break #Then we have reached all possible states in the game and thus break out of it 
A = allowableActions(testXWin)
#Probabilities are all set to 1 since every action 

humanLoop = True 

board =  [ ['*','*','*'],
                   ['*','*','*'],
                   ['*','*','*'],
                   ]




while(humanLoop): 
    
    #While our human loop is in the iteration we can first read in the value of the human  player as they go first 
    [i,j] = input("Please Enter a row and a column in the form of row,column").split(',') 
    board[int(i)][int(j)] = 'X' #Human will be player X 
    
    displayBoard(int(i),int(j),'X', board)
    
    #We determine if there is a win yet 
    calculateBoardWin(board,int(i),int(j),'X') 
    #calculateBoardWin(board,i,j,player)
    [i_computer, j_computer] = makeComputerMove(list_playerO,board)
    board[i_computer][j_computer] = 'O' #Computer will be player O 
    
    calculateBoardWin(board,i_computer,j_computer,'O')
    
    displayBoard(i_computer,j_computer,'O', board)
    
    
    