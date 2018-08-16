'''
Created on Aug 3, 2018

@author: Chris
'''
import numpy as np 
from _operator import length_hint
import random 
import copy 
import msvcrt
import matplotlib.pyplot as plt 
import numpy as np 

from math import fabs
import xml.etree.cElementTree as etree
import FileWriting
from tensorflow.python.training.device_util import current
from numpy.core.function_base import linspace
import statistics
#import XMLWriting
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
        #print("The game has been won by player ",player,"!")
        #We want to update the reward of the current state of the computer 
       
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
    epsilon = 0
    randNumber = random.uniform(0, 1)
    
    okayMove  = True 
    
    if randNumber > epsilon: 
        #Then we will choose a random move 
        
        #Although we must first make sure that the move is a valid move
        while(okayMove): 
            [i,j] = [random.randint(0,2),random.randint(0,2)]

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
       # displayBoard(i,j,player,board)
       
        
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
       # displayBoard(i,j,player,board)
        
        
        Q_primed = getQPrimed(board,states)
        updateQ(states, stateExists[0], Q_primed) #Now We need to update the function of the current state 
        
        calculateState(i, j, board, list_playerX,list_playerO, stateExists,player) #Check to see if O has won 
        stateNumber[0] = stateExists[0]
    
    
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
        #print("The game has been won by player ",player,"!")
        board = resetBoard(board)
    elif (isInList('*', board) == False): 
        board = resetBoard(board)
def kbfunc():
   x = msvcrt.kbhit()
   if x:
      ret = ord(msvcrt.getch())
   else:
      ret = 0
   return ret
def computeDeltadiff(previousReward, currentReward): 
    return 100 * fabs((currentReward - previousReward)/previousReward)
def gatherStatisticalData(statistics,currentPlayer): 
    #Add the latest rewar to the statistics class 
    I = currentPlayer.i 
    J = currentPlayer.j 
    statLength = len(statistics)
    currentReward = currentPlayer.rewards[I][J]
    
    if statLength != 0: 
        statistics.append(currentReward) #We append the current reward to the array which holds all of the statistical data
        
        previousRewardValue = statistics[(statLength) - 1]#Now using the iteration counter for the entire game we grab the previous reward 
        deltaDiff = computeDeltadiff(previousRewardValue, currentReward)
        
        if deltaDiff != 0: 
            
            print("Percentage Difference: ", deltaDiff,"%") 
            print("The current reward is", currentReward) 
            print("------------------")
            return deltaDiff #We compute the percent different between the two rewards in hopes of getting a threshold value that is less than some value for convergence 
        elif deltaDiff == 0: 
            #Then we may want to run through a few iterations to see how it plays out 
            print("Percentage Difference: ", deltaDiff,"%") 
            print("The current reward is", currentReward) 
            print("------------------")
            return deltaDiff
    elif statLength == 0: #Then our statlenghth is = 0 
        statistics.append(currentReward) #Append the current reward at least 
        
        return 100 #Return an abrturary value such as 100 % 
def createGraphicalChart(statistics,chartType):
    #These our all of our statistics for all of the  iteraions 
    if chartType == 1: 
        #Then our chart type is that of a scatter plot 
        XaxisLength = len(statistics) 
        x = np.linspace(0, XaxisLength, XaxisLength)
        plt.scatter(x, statistics)
        plt.show() 
    
    
gameLoop = True 


stateNumber_X = [0] 
stateNumber_O = [0]

list_playerX = initializeList() #Initialize the list of the first player 
list_playerO = initializeList() #Initialie the list of the second player 

totalReward = 0 

#Initialize the xml tree that will be used to store the XML file 

#This is essentiallly the training classifier to create the lookup table necessary to complete the proper training for the AI player 
iterationCount = 0 
iterationStats = []
testBoard = [['*','X','*'],['O','*','*'],['*','*','*']]

ConvergenceValue = 0.1 
while(gameLoop): 
    
    QlearnUpdate(list_playerX,stateNumber_X,board,'X',list_playerX,list_playerO) #Q learn update for the first AI player 

    #stateNumber += 1
    QlearnUpdate(list_playerO,stateNumber_O,board,'O',list_playerX,list_playerO) #Q learn update for for the second AI player 
    #print(stateNumber_O[0],stateNumber_X[0])
    if board == testBoard: 
        #here we need to take the state number of an arbiturary state and compute the reward delta 
        currentXPlayerState = list_playerX[stateNumber_X[0]]
        if gatherStatisticalData(iterationStats,currentXPlayerState) < ConvergenceValue: 
            print("You have reached the reward function convergence value less than  ", ConvergenceValue) #Reaching the end convergence value 
            
            break 
      
    #gate(previousReward, list_playerX[stateNumber_X[0]].rewards))
    iterationCount += 1 #This just counts the number of moves made by both player X and O 

createGraphicalChart(iterationStats, 1) #We want to create a graphical representation where 1 is a scatter chart 
    
#Probabilities are all set to 1 since every action 

humanLoop = True 
board =  [ ['*','*','*'],
                   ['*','*','*'],
                   ['*','*','*'],
                   ]

FileWriting.createXMLFILE(list_playerX) #This will take the current states, query them and output them to the correct XML file format     
while(humanLoop): 
    
    #While our human loop is in the iteration we can first read in the value of the human  player as they go first 
    [i,j] = input("Please Enter a row and a column in the form of row,column").split(',') 
    board[int(i)][int(j)] = 'X' #Human will be player X 
    
    #displayBoard(int(i),int(j),'X', board)
    
    #We determine if there is a win yet 
    calculateBoardWin(board,int(i),int(j),'X') 
    #calculateBoardWin(board,i,j,player)
    [i_computer, j_computer] = makeComputerMove(list_playerO,board)
    board[i_computer][j_computer] = 'O' #Computer will be player O 
    
    calculateBoardWin(board,i_computer,j_computer,'O')
    
    #displayBoard(i_computer,j_computer,'O', board)
    
    
    