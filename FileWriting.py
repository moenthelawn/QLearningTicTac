'''
Created on Aug 13, 2018

@author: Chris

'''
#from flowData import state
'''
This function will create the instance of the XML File Querying 
 
    def __init__(self,stateNumber,moves,rewards,board):
        self.stateNumber = stateNumber
        self.moves = [[1,1,1],[1,1,1],[1,1,1]]
        self.rewards = rewards 
        self.board = board 
        self.Q_max = 0 
        self.i = -1 #Max row action 
        self.j = -1  #Max column action 
        This is the original file format for the classes that is instantiated for the data storage 
'''

import xml.etree.cElementTree as etree

class State: 
    
    def __init__(self,stateNumber,moves,rewards,board):
        self.stateNumber = stateNumber
        self.moves = [[1,1,1],[1,1,1],[1,1,1]]
        self.rewards = rewards 
        self.board = board 
        self.Q_max = 10
        self.i = 0 #Max row action 
        self.j = 1  #Max column action 
        
    

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
def convertStringBoard(board):
    rows = len(board) 
    cols = len(board[0])
    BoardChar =''
    for i in range(rows): 
        for j in range(cols): 
            BoardChar += board[i][j]
    return BoardChar
            
            
def createXMLFILE(PlayerStates):
    root = etree.Element("root")
    doc = etree.SubElement(root, "Q LUT")
    statesNumber = len(PlayerStates) 

    for i in range(statesNumber): 
        #reward = PlayerStates[i].
        
        stateNumber = PlayerStates[i].stateNumber
        maxActionI = PlayerStates[i].i #We want to get the max action of that user's next move based on the board structure 
        maxActionJ = PlayerStates[i].j 
        
        '''State Number ----------------------------------------------'''
        stateSub = etree.SubElement(doc, "State", name="State Number")
        stateSub.text = str(stateNumber)#We input the state number into the array 
        
        strBoard = convertStringBoard(PlayerStates[i].board) 
        
        '''Max action corresponding to the row ''' 
        stateRow1 = etree.SubElement(stateSub,"Row",name="RowMove" )
        stateRow1.text = str(maxActionI) 
        
        '''Max action corresponding to the column'''
        stateRow2 = etree.SubElement(stateSub,"Col",name="ColMove")
        stateRow2.text = str(maxActionJ) 
        
        stateRow3 = etree.SubElement(stateSub,"Board",name="BoardConfig") 
        stateRow3.text = strBoard
        
        '''Board configuration''' 
        
    
    indent(root)    
    tree = etree.ElementTree(root)
    tree.write("filename.xml")
    
def indent(elem, level=0): #This function will create the correct indent for the XML format (so it lookspretty :)) 

    i = "\n" + level*"  "
    j = "\n" + (level-1)*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for subelem in elem:
            indent(subelem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = j
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = j
    return elem  


#s = State(1,None,[[3,1,5],[3,1,5],[3,1,5]],[['*','*','*'],['*','*','*'],['*','*','*']])
#s1 = State(2,None,[[3,1,5],[3,1,5],[3,1,5]],[['*','X','*'],['O','*','X'],['*','*','*']] )
#States = [s,s1]
#createXMLFILE(States)