# Connect-Four AI - A Battle Between AI/ML Algorithms
All I wanted to see the battle between AI/ML algorithms in the field of Connect-Four! Primary plan is to implement Connect-Four and some AI/ML algorithms like Minimax, Logistic Regression, CNN and see how they fight with each other. Shall we begin?

Wait! Check Connect-Four AI - A Battle Between AI Algorithms.ipynb for full implementation and output: https://github.com/hillolkallol/Connect-Four-AI-A-Battle-Between-AI-ML-Algorithms/blob/master/Connect-Four%20AI%20-%20A%20Battle%20Between%20AI%20Algorithms.ipynb

Now, it's time to start!

## What is Connect-Four?
___________________________________________________________________________________________________________
Connect-Four is a simple board game: https://www.youtube.com/watch?v=ylZBRUJi3UQ

Caution: Just for fun!! Haven't captured a few edge cases (i.e. wrong user input/invalid moves, index out of bound) Will be fixed in the next release! :p :p

Game Rules: Connect Four (also known as Four Up, Plot Four, Find Four, Four in a Row, Four in a Line, Drop Four, and Gravitrips in the Soviet Union) is a two-player connection board game, in which the players choose a color and then take turns dropping colored discs into a seven-column, six-row vertically suspended grid. The pieces fall straight down, occupying the lowest available space within the column. The objective of the game is to be the first to form a horizontal, vertical, or diagonal line of four of one's own discs. Connect Four is a solved game. The first player can always win by playing the right moves. Reference: Wikipedia- https://en.wikipedia.org/wiki/Connect_Four
___________________________________________________________________________________________________________

## Connect-Four Game Implementation
This game has two main characteristics- Move and Winner Check. CheckWinner method will be called after each move.

Before executing move, validMove method will be called from the client side just to check if the move is valid. If the move is valid, the move method will be called. The sign will be planted in the board (based on given row and col) Then the checkWinner method will be called. Last thing we need to check is if the match is a draw!

Next thing is to check if the player is the winner after the last move. Check-

* if four consecutive rows contain same sign OR
* if four consecutive columns contain same sign OR
* if four consecutive diagonal or anti-diagonal boxes contain same sign

### Next Move
Before executing move, validMove method will be called from the client side just to check if the move is valid.
If the move is valid, the move method will be called.
The sign will be planted in the board (based on given row and col)
Then the checkWinner method will be called.
Last thing we need to check is if the match is a draw!

```python
    def move (self, c, player):
        r = self.lastAvailableRow (c)
        # print (r, c)
        
        if r == -1:
            return r
        
        sign = self.players[player-1]
        
        self.board[r][c] = sign
        
        self.moves.append([r, c])
        
        if self.checkWinner (r, c, player):
            # print("Player ", player, " wins!")
            return 1
        
        if self.checkDraw ():
            # print("Draw!")
            return 0
        
        # print("Next move please!")
        return 2
```
### Check if the Match is a Draw
Check if the game is a draw. Check if the board is full.

```python
    def checkDraw (self):
        status = len(self.moves) == self.size * self.size
        if status:
            self.setWinner(0)
            
        return status
```

### Check if there is a Winner
Check if the player is the winner after the last move.
Check-
* if four consecutive rows contain same sign OR
* if four consecutive columns contain same sign OR
* if four consecutive diagonal or anti-diagonal boxes contain same sign

```python
    def checkWinner (self, r, c, player):
        status = self.checkRow (r, c, player) or self.checkCol (r, c, player) or self.checkDiagonal (r, c, player) or self.checkAntiDiagonal (r, c, player)
        
        if status:
            self.setWinner(player)
        return status

    def checkRow(self, r, c, player):
        count = 1
        currCol = c-1
        
        while currCol >= 0 and self.board[r][currCol] == self.players[player-1]:
            currCol -= 1
            count += 1
        
        currCol = c+1
        while currCol < self.size and self.board[r][currCol] == self.players[player-1]:
            currCol += 1
            count += 1
        
        return True if count >= 4 else False
        
    def checkCol(self, r, c, player):
        if self.size - r < 4:
            return False
        
        count = 1
        currRow = r+1
        while currRow < self.size and self.board[currRow][c] == self.players[player-1]:
            currRow += 1
            count += 1
        
        return True if count >= 4 else False
        
    def checkDiagonal (self, r, c, player):
        count = 1
        
        currR = r + 1
        currC = c + 1
        
        while currR < self.size and currC < self.size :
            if self.board[currR][currC] == self.players[player-1]:
                count += 1
            else:
                break
            currR += 1
            currC += 1
            
        currR = r - 1
        currC = c - 1
        
        while currR >= 0 and currC >= 0:
            if self.board[currR][currC] == self.players[player-1]:
                count += 1
            else:
                break
                
            currR -= 1
            currC -= 1
        
        return True if count >= 4 else False
        
    def checkAntiDiagonal (self, r, c, player):
        count = 1
        
        currR = r + 1
        currC = c - 1
        
        while currR < self.size and currC >= 0 :
            if self.board[currR][currC] == self.players[player-1]:
                count += 1
            else:
                break
            currR += 1
            currC -= 1
            
        currR = r - 1
        currC = c + 1
        
        while currR >= 0 and currC < self.size:
            if self.board[currR][currC] == self.players[player-1]:
                count += 1
            else:
                break
        
            currR -= 1
            currC += 1
        
        return True if count >= 4 else False
```

### Output of Normal Two Player Connect-Four Game
Below is the output of the normal two player Connect-Four game. Nothing fancy. Just wanted to check if my implementation works properly. And guess what? It works!!

```python
choose a board size: 
9
Pick your lucky color: R or B
B

  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  

Let's begin..
Player  1 's turn!
Choose a column betwwen 1 to  9
3

  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   | B |   |   |   |   |   |  

Player  2 's turn!
Choose a column betwwen 1 to  9
4

  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   | B | R |   |   |   |   |  

Player  1 's turn!
Choose a column betwwen 1 to  9
4

  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   | B |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   | B | R |   |   |   |   |  

Player  2 's turn!
Choose a column betwwen 1 to  9
2

  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   | B |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  | R | B | R |   |   |   |   |  

Player  1 's turn!
Choose a column betwwen 1 to  9
5

  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   | B |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  | R | B | R | B |   |   |   |  

Player  2 's turn!
Choose a column betwwen 1 to  9
5

  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   | B | R |   |   |   |  
- + - + - + - + - + - + - + - + -
  | R | B | R | B |   |   |   |  

Player  1 's turn!
Choose a column betwwen 1 to  9
5

  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   | B |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   | B | R |   |   |   |  
- + - + - + - + - + - + - + - + -
  | R | B | R | B |   |   |   |  

Player  2 's turn!
Choose a column betwwen 1 to  9
6

  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   | B |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   | B | R |   |   |   |  
- + - + - + - + - + - + - + - + -
  | R | B | R | B | R |   |   |  

Player  1 's turn!
Choose a column betwwen 1 to  9
6

  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   | B |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   | B | R | B |   |   |  
- + - + - + - + - + - + - + - + -
  | R | B | R | B | R |   |   |  

Player  2 's turn!
Choose a column betwwen 1 to  9
6

  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   | B | R |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   | B | R | B |   |   |  
- + - + - + - + - + - + - + - + -
  | R | B | R | B | R |   |   |  

Player  1 's turn!
Choose a column betwwen 1 to  9
6
Player  1  wins!

  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   |   |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   |   | B |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   |   | B | R |   |   |  
- + - + - + - + - + - + - + - + -
  |   |   | B | R | B |   |   |  
- + - + - + - + - + - + - + - + -
  | R | B | R | B | R |   |   |  

===============
One more? (y/n)
===============
n
```

## Connect-Four AI
Extending the basic implementation of Connect-Four and adding necessary features to make it compitable with AI system. 

```python
class ConnectFourAI (ConnectFour):
    
    def __init__(self, size, players):
        super().__init__(size, players)
        
    def allPossibleNextMoves (self):
        possibleMoves = []
        
        for col in range(self.size):
            if (self.isValidMove(col)):
                possibleMoves.append(col)
        
        return possibleMoves
```

## MiniMax Algorithm
Wiki: Minimax (sometimes MinMax, MM[1] or saddle point[2]) is a decision rule used in artificial intelligence, decision theory, game theory, statistics, and philosophy for minimizing the possible loss for a worst case (maximum loss) scenario. When dealing with gains, it is referred to as "maximin"—to maximize the minimum gain. Originally formulated for n-player zero-sum game theory, covering both the cases where players take alternate moves and those where they make simultaneous moves, it has also been extended to more complex games and to general decision-making in the presence of uncertainty.
Source: https://en.wikipedia.org/wiki/Minimax

### Pseudocode for Wikipedia
Pseudocode of basic minimax algorithm is given below-

```python
function minimax(node, depth, maximizingPlayer) is
    if depth = 0 or node is a terminal node then
        return the heuristic value of node
    if maximizingPlayer then
        value := −∞
        for each child of node do
            value := max(value, minimax(child, depth − 1, FALSE))
        return value
    else (* minimizing player *)
        value := +∞
        for each child of node do
            value := min(value, minimax(child, depth − 1, TRUE))
        return value
```

Basic MiniMax implemtation is given below. Didn't implement Alpha-Beta Pruning.

Thanks to wiki,The Coding Train, levelup and many other online resources to help me understanding and implementing MiniMax: 
https://en.wikipedia.org/wiki/Minimax
https://levelup.gitconnected.com/mastering-Connect-Four-with-minimax-algorithm-3394d65fa88f
https://www.youtube.com/watch?v=trKjYdBASyQ&t=1196s

### Find Best Possible Move
Iterate through all the possible moves and call minimax each time to find the best possible move

```python
    def findBestMiniMaxMove (self, player):
        bestScore = -math.inf
        bestMove = None
        counter = [0]
        
        for possibleMove in self.game.allPossibleNextMoves():
            self.game.move(possibleMove, player)
            score = self.minimax (False, player, 0, counter)
            self.game.undo()

            if (score > bestScore):
                bestScore = score
                bestMove = possibleMove
        
        print ("I have compared ", counter[0], " combinations and executed the best move out of it. I can't lose, dude!")
        return bestMove
```

### Minimax
Return Max Score and Min Score respectively for Maximizing and Minimizing player. Limited the depth to make it computationally efficient.

```python
    def minimax (self, isMax, player, depth, counter):
        
        if depth == 5:
            return 0
        
        counter[0] = counter[0] + 1
        
        winner = self.game.getWinner()
        if (not (winner == None)):
            if (winner == 0):
                return 0
            elif (winner == player):
                return 10 - depth
            else:
                return depth - 10
        
        maxScore = -math.inf
        minScore = math.inf
        
        for possibleMove in self.game.allPossibleNextMoves():
            currPlayer = player if isMax else 2 if (player == 1) else 1
            
            self.game.move(possibleMove, currPlayer)
            score = self.minimax (not isMax, player, depth + 1, counter)
            self.game.undo()
            
            if (score > maxScore):
                maxScore = score
            if (score < minScore):
                minScore = score
        
        return maxScore if isMax else minScore
```

## Generating DataSet
We need a good dataset before leveraging machine learning algorithms. There are many ways to generate data. For my model, I generated data after each move. 

### Generate New Row
Add new row to temporary dataset (newData) after every move. 
This is using to generate new data so that we can train our machine learning model with new data after each game.

This function creates two rows for each move considering both of the players as winner 

```python
    def generateNewRow(self):
        newRow = []
        for row in range (self.size):
            for col in range (self.size):
                val = 0
                if (self.board[row][col] == self.players[0]):
                    val = 1
                elif (self.board[row][col] == self.players[1]):
                    val = -1
                
                newRow.append(val)
        
        newInvertRow = [v if v == 0 else -1 if v == 1 else 1 for v in newRow]
        
        self.newData.append (newRow)
        self.newData.append (newInvertRow)
```

### Get New Data
GetNewData will be called at the end of each match. 
This function labels all the data and return a new set of dataset so that we can train our ML model with this new set of data

```python
    def getNewData (self, winner):
        if (winner == 1): 
            newTrainY = [1 if i % 2 == 0 else 2 for i in range(len(self.newData))]
        elif (winner == 2): 
            newTrainY = [2 if i % 2 == 0 else 1 for i in range(len(self.newData))]
        else: 
            newTrainY = [0 for i in range(len(self.newData))]
        
        newTrainX = self.newData
        self.newData = []
        
        print("Size of newTrainX and newTrainY: ", len(newTrainX), len(newTrainY))
        # print(newTrainX)
        # print(newTrainY)
        
        return newTrainX, newTrainY
```

## Logistic Regression Algorithm
Logistic Regression is well known machine learning algorithm. Not going to detail of this algorithm.

### Find Best Possible Move
Iterate through all the possible moves, generate new test data and call logistic regression algorithm to find the best possible move based on the probability of winning.

```python
    def findBestLogisticMove (self, player):
        testX = []
        positions = []
        
        for possibleMove in self.game.allPossibleNextMoves():
            self.game.move(possibleMove, player)
            positions.append (possibleMove)
            testX.append (self.generateTestX())
            self.game.undo()
        
        index = 1 if player == 1 else 2
        
        predictions = np.around(self.logisticRegressionTesting (testX), decimals=2)
        
        maxProb = np.amax(predictions[:, index])
        moveIndex = np.where(predictions[:, index] == maxProb)[0][0]

        return positions[moveIndex]
```

### Logistic Regression Training
Train your logistic Regression model with the new and old dataset

```python
    dataset = np.concatenate((np.asarray(self.trainX), np.asarray([self.trainY]).T), axis=1)
        np.random.shuffle(dataset)
        
        X = dataset[:, :-1]
        y = dataset[:, -1]
        self.LRModel = LogisticRegression(random_state=0).fit(X, y)
```

### Logistic Regression Testing
Test your logistic Regression model with all possible moves and find the best move based on the probability of winning.

```python
    def logisticRegressionTesting (self, testX):
        return self.LRModel.predict_proba(np.asarray(testX))
```

### Generate Test Data
Generating testing data based on all possible moves.

```python
     def generateTestX (self):
        newRow = []
        for row in range (self.game.size):
            for col in range (self.game.size):
                val = 0
                if (self.game.board[row][col] == self.game.players[0]):
                    val = 1
                elif (self.game.board[row][col] == self.game.players[1]):
                    val = -1
                
                newRow.append(val)
        return newRow
```

## ConvNet Algorithm
ConvNet is well known deep learning model. Decided to not to discuss about it here. Moving on..

### Find Best Possible Move
Iterate through all the possible moves, generate new test data and call ConvNet algorithm to find the best possible move based on the probability of winning.

```python
    def findBestCNNMove (self, player):
        testX = []
        positions = []
        accuracy = []
        
        desireClass = 1 if player == 1 else 2
        
        for possibleMove in self.game.allPossibleNextMoves():
            self.game.move(possibleMove, player)
            positions.append (possibleMove)
            test_loss, test_acc = self.convolutionalNeuralNetworkTesting ([np.asarray(self.generateTestX()).reshape(self.game.size, self.game.size, 1)], [desireClass])
            accuracy.append (test_acc)
            self.game.undo()
        
        # print(accuracy)
        maxProb = np.amax(accuracy)
        # print(maxProb)
        moveIndex = np.where(accuracy == maxProb)[0][0]
        # print(moveIndex)

        return positions[moveIndex]
```

### ConvNet Training
Here is the architecture of my ConvNet-

```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_4 (Conv2D)            (None, 3, 3, 32)          160       
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 2, 2, 32)          0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 2, 2, 64)          8256      
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 1, 1, 64)          0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 1, 1, 32)          8224      
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 1, 1, 32)          0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 1, 1, 16)          2064      
_________________________________________________________________
flatten_1 (Flatten)          (None, 16)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 32)                544       
_________________________________________________________________
dense_3 (Dense)              (None, 3)                 99        
=================================================================
Total params: 19,347
Trainable params: 19,347
Non-trainable params: 0
_________________________________________________________________

```

Please note that, I haven't done any hyperparameter tuning, which is, I know, not a good practice. My all I wanted was making my hand dirty in a small board game so that I can use it later in large board.

Train your CNN model with the new and old dataset. 

```python
    def convolutionalNeuralNetworkTraining (self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (2, 2), activation='relu', padding="same", input_shape=(self.game.size, self.game.size, 1)))
        self.model.add(layers.MaxPooling2D((2, 2), padding="same")) # dim_ordering="th"
        self.model.add(layers.Conv2D(64, (2, 2), activation='relu', padding="same"))
        self.model.add(layers.MaxPooling2D((2, 2), padding="same"))
        self.model.add(layers.Conv2D(32, (2, 2), activation='relu', padding="same"))
        self.model.add(layers.MaxPooling2D((2, 2), padding="same"))
        self.model.add(layers.Conv2D(16, (2, 2), activation='relu', padding="same"))
        
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(32, activation='relu'))
        self.model.add(layers.Dense(3, activation = "softmax"))
        
        self.model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        
        print(self.model.summary())
        
        history = self.model.fit(np.asarray(self.trainX), np.asarray(self.trainY), epochs=30, shuffle=True)
```

### ConvNet Testing
Test your ConvNet model with all possible moves and find the best move based on the probability of winning.

```python
    def convolutionalNeuralNetworkTesting (self, test_images,  test_labels):
        return self.model.evaluate(np.asarray(test_images),  np.asarray(test_labels), verbose=0)
```

### Battle 1: Human Vs Minimax Algorithm!
Below is the output of Human Vs Minimax!. Quess what? Minimax doesn't lose!

```python
choose a board size: 
5
Pick your lucky color: R or B
R

  |   |   |   |  
- + - + - + - + -
  |   |   |   |  
- + - + - + - + -
  |   |   |   |  
- + - + - + - + -
  |   |   |   |  
- + - + - + - + -
  |   |   |   |  

Let's begin..
It's MiniMax's turn!
I have compared  97505  combinations and executed the best move out of it. I can't lose, dude!

  |   |   |   |  
- + - + - + - + -
  |   |   |   |  
- + - + - + - + -
  |   |   |   |  
- + - + - + - + -
  |   |   |   |  
- + - + - + - + -
B |   |   |   |  

It's human's turn!
Choose a number betwwen 1 to  5
2

  |   |   |   |  
- + - + - + - + -
  |   |   |   |  
- + - + - + - + -
  |   |   |   |  
- + - + - + - + -
  |   |   |   |  
- + - + - + - + -
B | R |   |   |  

It's MiniMax's turn!
I have compared  95314  combinations and executed the best move out of it. I can't lose, dude!

  |   |   |   |  
- + - + - + - + -
  |   |   |   |  
- + - + - + - + -
  |   |   |   |  
- + - + - + - + -
B |   |   |   |  
- + - + - + - + -
B | R |   |   |  

It's human's turn!
Choose a number betwwen 1 to  5
2

  |   |   |   |  
- + - + - + - + -
  |   |   |   |  
- + - + - + - + -
  |   |   |   |  
- + - + - + - + -
B | R |   |   |  
- + - + - + - + -
B | R |   |   |  

It's MiniMax's turn!
I have compared  80821  combinations and executed the best move out of it. I can't lose, dude!

  |   |   |   |  
- + - + - + - + -
  |   |   |   |  
- + - + - + - + -
B |   |   |   |  
- + - + - + - + -
B | R |   |   |  
- + - + - + - + -
B | R |   |   |  

It's human's turn!
Choose a number betwwen 1 to  5
1

  |   |   |   |  
- + - + - + - + -
R |   |   |   |  
- + - + - + - + -
B |   |   |   |  
- + - + - + - + -
B | R |   |   |  
- + - + - + - + -
B | R |   |   |  

It's MiniMax's turn!
I have compared  51468  combinations and executed the best move out of it. I can't lose, dude!

B |   |   |   |  
- + - + - + - + -
R |   |   |   |  
- + - + - + - + -
B |   |   |   |  
- + - + - + - + -
B | R |   |   |  
- + - + - + - + -
B | R |   |   |  

It's human's turn!
Choose a number betwwen 1 to  5
5

B |   |   |   |  
- + - + - + - + -
R |   |   |   |  
- + - + - + - + -
B |   |   |   |  
- + - + - + - + -
B | R |   |   |  
- + - + - + - + -
B | R |   |   | R

It's MiniMax's turn!
I have compared  16931  combinations and executed the best move out of it. I can't lose, dude!

B |   |   |   |  
- + - + - + - + -
R |   |   |   |  
- + - + - + - + -
B | B |   |   |  
- + - + - + - + -
B | R |   |   |  
- + - + - + - + -
B | R |   |   | R

It's human's turn!
Choose a number betwwen 1 to  5
5

B |   |   |   |  
- + - + - + - + -
R |   |   |   |  
- + - + - + - + -
B | B |   |   |  
- + - + - + - + -
B | R |   |   | R
- + - + - + - + -
B | R |   |   | R

It's MiniMax's turn!
I have compared  12610  combinations and executed the best move out of it. I can't lose, dude!

B |   |   |   |  
- + - + - + - + -
R | B |   |   |  
- + - + - + - + -
B | B |   |   |  
- + - + - + - + -
B | R |   |   | R
- + - + - + - + -
B | R |   |   | R

It's human's turn!
Choose a number betwwen 1 to  5
5

B |   |   |   |  
- + - + - + - + -
R | B |   |   |  
- + - + - + - + -
B | B |   |   | R
- + - + - + - + -
B | R |   |   | R
- + - + - + - + -
B | R |   |   | R

It's MiniMax's turn!
I have compared  3658  combinations and executed the best move out of it. I can't lose, dude!

B |   |   |   |  
- + - + - + - + -
R | B |   |   | B
- + - + - + - + -
B | B |   |   | R
- + - + - + - + -
B | R |   |   | R
- + - + - + - + -
B | R |   |   | R

It's human's turn!
Choose a number betwwen 1 to  5
3

B |   |   |   |  
- + - + - + - + -
R | B |   |   | B
- + - + - + - + -
B | B |   |   | R
- + - + - + - + -
B | R |   |   | R
- + - + - + - + -
B | R | R |   | R

It's MiniMax's turn!
I have compared  1796  combinations and executed the best move out of it. I can't lose, dude!

B |   |   |   |  
- + - + - + - + -
R | B |   |   | B
- + - + - + - + -
B | B |   |   | R
- + - + - + - + -
B | R |   |   | R
- + - + - + - + -
B | R | R | B | R

It's human's turn!
Choose a number betwwen 1 to  5
4

B |   |   |   |  
- + - + - + - + -
R | B |   |   | B
- + - + - + - + -
B | B |   |   | R
- + - + - + - + -
B | R |   | R | R
- + - + - + - + -
B | R | R | B | R

It's MiniMax's turn!
I have compared  1597  combinations and executed the best move out of it. I can't lose, dude!

B |   |   |   |  
- + - + - + - + -
R | B |   |   | B
- + - + - + - + -
B | B |   |   | R
- + - + - + - + -
B | R | B | R | R
- + - + - + - + -
B | R | R | B | R

It's human's turn!
Choose a number betwwen 1 to  5
4

B |   |   |   |  
- + - + - + - + -
R | B |   |   | B
- + - + - + - + -
B | B |   | R | R
- + - + - + - + -
B | R | B | R | R
- + - + - + - + -
B | R | R | B | R

It's MiniMax's turn!
I have compared  771  combinations and executed the best move out of it. I can't lose, dude!

B | B |   |   |  
- + - + - + - + -
R | B |   |   | B
- + - + - + - + -
B | B |   | R | R
- + - + - + - + -
B | R | B | R | R
- + - + - + - + -
B | R | R | B | R

It's human's turn!
Choose a number betwwen 1 to  5
4

B | B |   |   |  
- + - + - + - + -
R | B |   | R | B
- + - + - + - + -
B | B |   | R | R
- + - + - + - + -
B | R | B | R | R
- + - + - + - + -
B | R | R | B | R

It's MiniMax's turn!
I have compared  44  combinations and executed the best move out of it. I can't lose, dude!

B | B |   | B |  
- + - + - + - + -
R | B |   | R | B
- + - + - + - + -
B | B |   | R | R
- + - + - + - + -
B | R | B | R | R
- + - + - + - + -
B | R | R | B | R

It's human's turn!
Choose a number betwwen 1 to  5
3

B | B |   | B |  
- + - + - + - + -
R | B |   | R | B
- + - + - + - + -
B | B | R | R | R
- + - + - + - + -
B | R | B | R | R
- + - + - + - + -
B | R | R | B | R

It's MiniMax's turn!
I have compared  4  combinations and executed the best move out of it. I can't lose, dude!

B | B |   | B |  
- + - + - + - + -
R | B | B | R | B
- + - + - + - + -
B | B | R | R | R
- + - + - + - + -
B | R | B | R | R
- + - + - + - + -
B | R | R | B | R

MiniMax wins!
```

### Battle 2: Minimax Vs Minimax!
Below is the output of Minimax Vs Minimax!. No one wins!

```python
Let's begin..

  |   |   |   |   |  
- + - + - + - + - + -
  |   |   |   |   |  
- + - + - + - + - + -
  |   |   |   |   |  
- + - + - + - + - + -
  |   |   |   |   |  
- + - + - + - + - + -
  |   |   |   |   |  
- + - + - + - + - + -
  |   |   |   |   |  

It's MiniMax1's turn!
I have compared  9330  combinations and executed the best move out of it. I can't lose, dude!

  |   |   |   |   |  
- + - + - + - + - + -
  |   |   |   |   |  
- + - + - + - + - + -
  |   |   |   |   |  
- + - + - + - + - + -
  |   |   |   |   |  
- + - + - + - + - + -
  |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  

It's MiniMax2's turn!
I have compared  9330  combinations and executed the best move out of it. I can't lose, dude!

  |   |   |   |   |  
- + - + - + - + - + -
  |   |   |   |   |  
- + - + - + - + - + -
  |   |   |   |   |  
- + - + - + - + - + -
  |   |   |   |   |  
- + - + - + - + - + -
O |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  

It's MiniMax1's turn!
I have compared  9329  combinations and executed the best move out of it. I can't lose, dude!

  |   |   |   |   |  
- + - + - + - + - + -
  |   |   |   |   |  
- + - + - + - + - + -
  |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
O |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  

It's MiniMax2's turn!
I have compared  9303  combinations and executed the best move out of it. I can't lose, dude!

  |   |   |   |   |  
- + - + - + - + - + -
  |   |   |   |   |  
- + - + - + - + - + -
O |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
O |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  

It's MiniMax1's turn!
I have compared  9032  combinations and executed the best move out of it. I can't lose, dude!

  |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
O |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
O |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  

It's MiniMax2's turn!
I have compared  7616  combinations and executed the best move out of it. I can't lose, dude!

O |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
O |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
O |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  

It's MiniMax1's turn!
I have compared  3905  combinations and executed the best move out of it. I can't lose, dude!

O |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
O |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
O |   |   |   |   |  
- + - + - + - + - + -
X | X |   |   |   |  

It's MiniMax2's turn!
I have compared  3785  combinations and executed the best move out of it. I can't lose, dude!

O |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
O |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
O | O |   |   |   |  
- + - + - + - + - + -
X | X |   |   |   |  

It's MiniMax1's turn!
I have compared  3664  combinations and executed the best move out of it. I can't lose, dude!

O |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
O |   |   |   |   |  
- + - + - + - + - + -
X | X |   |   |   |  
- + - + - + - + - + -
O | O |   |   |   |  
- + - + - + - + - + -
X | X |   |   |   |  

It's MiniMax2's turn!
I have compared  3763  combinations and executed the best move out of it. I can't lose, dude!

O |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
O | O |   |   |   |  
- + - + - + - + - + -
X | X |   |   |   |  
- + - + - + - + - + -
O | O |   |   |   |  
- + - + - + - + - + -
X | X |   |   |   |  

It's MiniMax1's turn!
I have compared  3468  combinations and executed the best move out of it. I can't lose, dude!

O |   |   |   |   |  
- + - + - + - + - + -
X | X |   |   |   |  
- + - + - + - + - + -
O | O |   |   |   |  
- + - + - + - + - + -
X | X |   |   |   |  
- + - + - + - + - + -
O | O |   |   |   |  
- + - + - + - + - + -
X | X |   |   |   |  

It's MiniMax2's turn!
I have compared  2857  combinations and executed the best move out of it. I can't lose, dude!

O | O |   |   |   |  
- + - + - + - + - + -
X | X |   |   |   |  
- + - + - + - + - + -
O | O |   |   |   |  
- + - + - + - + - + -
X | X |   |   |   |  
- + - + - + - + - + -
O | O |   |   |   |  
- + - + - + - + - + -
X | X |   |   |   |  

It's MiniMax1's turn!
I have compared  1244  combinations and executed the best move out of it. I can't lose, dude!

O | O |   |   |   |  
- + - + - + - + - + -
X | X |   |   |   |  
- + - + - + - + - + -
O | O |   |   |   |  
- + - + - + - + - + -
X | X |   |   |   |  
- + - + - + - + - + -
O | O |   |   |   |  
- + - + - + - + - + -
X | X | X |   |   |  

It's MiniMax2's turn!
I have compared  1004  combinations and executed the best move out of it. I can't lose, dude!

O | O |   |   |   |  
- + - + - + - + - + -
X | X |   |   |   |  
- + - + - + - + - + -
O | O |   |   |   |  
- + - + - + - + - + -
X | X |   |   |   |  
- + - + - + - + - + -
O | O |   |   |   |  
- + - + - + - + - + -
X | X | X | O |   |  

It's MiniMax1's turn!
I have compared  1316  combinations and executed the best move out of it. I can't lose, dude!

O | O |   |   |   |  
- + - + - + - + - + -
X | X |   |   |   |  
- + - + - + - + - + -
O | O |   |   |   |  
- + - + - + - + - + -
X | X |   |   |   |  
- + - + - + - + - + -
O | O | X |   |   |  
- + - + - + - + - + -
X | X | X | O |   |  

It's MiniMax2's turn!
I have compared  1295  combinations and executed the best move out of it. I can't lose, dude!

O | O |   |   |   |  
- + - + - + - + - + -
X | X |   |   |   |  
- + - + - + - + - + -
O | O |   |   |   |  
- + - + - + - + - + -
X | X | O |   |   |  
- + - + - + - + - + -
O | O | X |   |   |  
- + - + - + - + - + -
X | X | X | O |   |  

It's MiniMax1's turn!
I have compared  1295  combinations and executed the best move out of it. I can't lose, dude!

O | O |   |   |   |  
- + - + - + - + - + -
X | X |   |   |   |  
- + - + - + - + - + -
O | O | X |   |   |  
- + - + - + - + - + -
X | X | O |   |   |  
- + - + - + - + - + -
O | O | X |   |   |  
- + - + - + - + - + -
X | X | X | O |   |  

It's MiniMax2's turn!
I have compared  1126  combinations and executed the best move out of it. I can't lose, dude!

O | O |   |   |   |  
- + - + - + - + - + -
X | X | O |   |   |  
- + - + - + - + - + -
O | O | X |   |   |  
- + - + - + - + - + -
X | X | O |   |   |  
- + - + - + - + - + -
O | O | X |   |   |  
- + - + - + - + - + -
X | X | X | O |   |  

It's MiniMax1's turn!
I have compared  876  combinations and executed the best move out of it. I can't lose, dude!

O | O | X |   |   |  
- + - + - + - + - + -
X | X | O |   |   |  
- + - + - + - + - + -
O | O | X |   |   |  
- + - + - + - + - + -
X | X | O |   |   |  
- + - + - + - + - + -
O | O | X |   |   |  
- + - + - + - + - + -
X | X | X | O |   |  

It's MiniMax2's turn!
I have compared  315  combinations and executed the best move out of it. I can't lose, dude!

O | O | X |   |   |  
- + - + - + - + - + -
X | X | O |   |   |  
- + - + - + - + - + -
O | O | X |   |   |  
- + - + - + - + - + -
X | X | O |   |   |  
- + - + - + - + - + -
O | O | X | O |   |  
- + - + - + - + - + -
X | X | X | O |   |  

It's MiniMax1's turn!
I have compared  248  combinations and executed the best move out of it. I can't lose, dude!

O | O | X |   |   |  
- + - + - + - + - + -
X | X | O |   |   |  
- + - + - + - + - + -
O | O | X |   |   |  
- + - + - + - + - + -
X | X | O |   |   |  
- + - + - + - + - + -
O | O | X | O |   |  
- + - + - + - + - + -
X | X | X | O | X |  

It's MiniMax2's turn!
I have compared  323  combinations and executed the best move out of it. I can't lose, dude!

O | O | X |   |   |  
- + - + - + - + - + -
X | X | O |   |   |  
- + - + - + - + - + -
O | O | X |   |   |  
- + - + - + - + - + -
X | X | O | O |   |  
- + - + - + - + - + -
O | O | X | O |   |  
- + - + - + - + - + -
X | X | X | O | X |  

It's MiniMax1's turn!
I have compared  251  combinations and executed the best move out of it. I can't lose, dude!

O | O | X |   |   |  
- + - + - + - + - + -
X | X | O |   |   |  
- + - + - + - + - + -
O | O | X | X |   |  
- + - + - + - + - + -
X | X | O | O |   |  
- + - + - + - + - + -
O | O | X | O |   |  
- + - + - + - + - + -
X | X | X | O | X |  

It's MiniMax2's turn!
I have compared  302  combinations and executed the best move out of it. I can't lose, dude!

O | O | X |   |   |  
- + - + - + - + - + -
X | X | O | O |   |  
- + - + - + - + - + -
O | O | X | X |   |  
- + - + - + - + - + -
X | X | O | O |   |  
- + - + - + - + - + -
O | O | X | O |   |  
- + - + - + - + - + -
X | X | X | O | X |  

It's MiniMax1's turn!
I have compared  191  combinations and executed the best move out of it. I can't lose, dude!

O | O | X | X |   |  
- + - + - + - + - + -
X | X | O | O |   |  
- + - + - + - + - + -
O | O | X | X |   |  
- + - + - + - + - + -
X | X | O | O |   |  
- + - + - + - + - + -
O | O | X | O |   |  
- + - + - + - + - + -
X | X | X | O | X |  

It's MiniMax2's turn!
I have compared  62  combinations and executed the best move out of it. I can't lose, dude!

O | O | X | X |   |  
- + - + - + - + - + -
X | X | O | O |   |  
- + - + - + - + - + -
O | O | X | X |   |  
- + - + - + - + - + -
X | X | O | O |   |  
- + - + - + - + - + -
O | O | X | O | O |  
- + - + - + - + - + -
X | X | X | O | X |  

It's MiniMax1's turn!
I have compared  59  combinations and executed the best move out of it. I can't lose, dude!

O | O | X | X |   |  
- + - + - + - + - + -
X | X | O | O |   |  
- + - + - + - + - + -
O | O | X | X |   |  
- + - + - + - + - + -
X | X | O | O | X |  
- + - + - + - + - + -
O | O | X | O | O |  
- + - + - + - + - + -
X | X | X | O | X |  

It's MiniMax2's turn!
I have compared  55  combinations and executed the best move out of it. I can't lose, dude!

O | O | X | X |   |  
- + - + - + - + - + -
X | X | O | O |   |  
- + - + - + - + - + -
O | O | X | X | O |  
- + - + - + - + - + -
X | X | O | O | X |  
- + - + - + - + - + -
O | O | X | O | O |  
- + - + - + - + - + -
X | X | X | O | X |  

It's MiniMax1's turn!
I have compared  40  combinations and executed the best move out of it. I can't lose, dude!

O | O | X | X |   |  
- + - + - + - + - + -
X | X | O | O | X |  
- + - + - + - + - + -
O | O | X | X | O |  
- + - + - + - + - + -
X | X | O | O | X |  
- + - + - + - + - + -
O | O | X | O | O |  
- + - + - + - + - + -
X | X | X | O | X |  

It's MiniMax2's turn!
I have compared  20  combinations and executed the best move out of it. I can't lose, dude!

O | O | X | X | O |  
- + - + - + - + - + -
X | X | O | O | X |  
- + - + - + - + - + -
O | O | X | X | O |  
- + - + - + - + - + -
X | X | O | O | X |  
- + - + - + - + - + -
O | O | X | O | O |  
- + - + - + - + - + -
X | X | X | O | X |  

It's MiniMax1's turn!
I have compared  5  combinations and executed the best move out of it. I can't lose, dude!

O | O | X | X | O |  
- + - + - + - + - + -
X | X | O | O | X |  
- + - + - + - + - + -
O | O | X | X | O |  
- + - + - + - + - + -
X | X | O | O | X |  
- + - + - + - + - + -
O | O | X | O | O |  
- + - + - + - + - + -
X | X | X | O | X | X

It's MiniMax2's turn!
I have compared  5  combinations and executed the best move out of it. I can't lose, dude!

O | O | X | X | O |  
- + - + - + - + - + -
X | X | O | O | X |  
- + - + - + - + - + -
O | O | X | X | O |  
- + - + - + - + - + -
X | X | O | O | X |  
- + - + - + - + - + -
O | O | X | O | O | O
- + - + - + - + - + -
X | X | X | O | X | X

It's MiniMax1's turn!
I have compared  4  combinations and executed the best move out of it. I can't lose, dude!

O | O | X | X | O |  
- + - + - + - + - + -
X | X | O | O | X |  
- + - + - + - + - + -
O | O | X | X | O |  
- + - + - + - + - + -
X | X | O | O | X | X
- + - + - + - + - + -
O | O | X | O | O | O
- + - + - + - + - + -
X | X | X | O | X | X

It's MiniMax2's turn!
I have compared  3  combinations and executed the best move out of it. I can't lose, dude!

O | O | X | X | O |  
- + - + - + - + - + -
X | X | O | O | X |  
- + - + - + - + - + -
O | O | X | X | O | O
- + - + - + - + - + -
X | X | O | O | X | X
- + - + - + - + - + -
O | O | X | O | O | O
- + - + - + - + - + -
X | X | X | O | X | X

It's MiniMax1's turn!
I have compared  2  combinations and executed the best move out of it. I can't lose, dude!

O | O | X | X | O |  
- + - + - + - + - + -
X | X | O | O | X | X
- + - + - + - + - + -
O | O | X | X | O | O
- + - + - + - + - + -
X | X | O | O | X | X
- + - + - + - + - + -
O | O | X | O | O | O
- + - + - + - + - + -
X | X | X | O | X | X

It's MiniMax2's turn!
I have compared  1  combinations and executed the best move out of it. I can't lose, dude!

O | O | X | X | O | O
- + - + - + - + - + -
X | X | O | O | X | X
- + - + - + - + - + -
O | O | X | X | O | O
- + - + - + - + - + -
X | X | O | O | X | X
- + - + - + - + - + -
O | O | X | O | O | O
- + - + - + - + - + -
X | X | X | O | X | X

Match Draw!
```

### Battle 3: Minimax Vs Logistic Regression!
Below is the output of Minimax Vs Logistic Regression!. Minimax is always unbeatable. Logistic Regression is unbeatable as well if it is trained with good dataset. Logistic Regression is way faster than Minimax (obviously minimax would perform faster with alpha-beta prouning.) Logistic Regression can lose first few matches if the model is not trained well, but we are training the model at the end of each match, so logistic regression will definitely make a come back! Fingers Crossed!! 

Data has been gereated by placing some random moves against minimax algorithm.

```python
======================== IT'S BATTLE TIME =================================
Battle number:  1
==============================================================
Minimax:  0 Logistic Regression:  0 Draw:  0
==============================================================
It's Logistic Regression's turn!

  |   |   |   |   |  
- + - + - + - + - + -
  |   |   |   |   |  
- + - + - + - + - + -
  |   |   |   |   |  
- + - + - + - + - + -
  |   |   |   |   |  
- + - + - + - + - + -
  |   |   |   |   |  
- + - + - + - + - + -
O |   |   |   |   |  

It's MiniMax's turn!
I have compared  9330  combinations and executed the best move out of it. I can't lose, dude!

  |   |   |   |   |  
- + - + - + - + - + -
  |   |   |   |   |  
- + - + - + - + - + -
  |   |   |   |   |  
- + - + - + - + - + -
  |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
O |   |   |   |   |  

It's Logistic Regression's turn!

  |   |   |   |   |  
- + - + - + - + - + -
  |   |   |   |   |  
- + - + - + - + - + -
  |   |   |   |   |  
- + - + - + - + - + -
O |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
O |   |   |   |   |  

It's MiniMax's turn!
I have compared  9303  combinations and executed the best move out of it. I can't lose, dude!

  |   |   |   |   |  
- + - + - + - + - + -
  |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
O |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
O |   |   |   |   |  

It's Logistic Regression's turn!

  |   |   |   |   |  
- + - + - + - + - + -
  |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
O |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
O |   |   |   |   | O

It's MiniMax's turn!
I have compared  9032  combinations and executed the best move out of it. I can't lose, dude!

  |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
O |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
O |   |   |   |   | O

It's Logistic Regression's turn!

O |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
O |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
O |   |   |   |   | O

It's MiniMax's turn!
I have compared  3905  combinations and executed the best move out of it. I can't lose, dude!

O |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
O |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
O | X |   |   |   | O

It's Logistic Regression's turn!

O |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
O |   |   |   |   |  
- + - + - + - + - + -
X | O |   |   |   |  
- + - + - + - + - + -
O | X |   |   |   | O

It's MiniMax's turn!
I have compared  3904  combinations and executed the best move out of it. I can't lose, dude!

O |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
O | X |   |   |   |  
- + - + - + - + - + -
X | O |   |   |   |  
- + - + - + - + - + -
O | X |   |   |   | O

It's Logistic Regression's turn!

O |   |   |   |   |  
- + - + - + - + - + -
X |   |   |   |   |  
- + - + - + - + - + -
X | O |   |   |   |  
- + - + - + - + - + -
O | X |   |   |   |  
- + - + - + - + - + -
X | O |   |   |   |  
- + - + - + - + - + -
O | X |   |   |   | O

It's MiniMax's turn!
I have compared  3676  combinations and executed the best move out of it. I can't lose, dude!

O |   |   |   |   |  
- + - + - + - + - + -
X | X |   |   |   |  
- + - + - + - + - + -
X | O |   |   |   |  
- + - + - + - + - + -
O | X |   |   |   |  
- + - + - + - + - + -
X | O |   |   |   |  
- + - + - + - + - + -
O | X |   |   |   | O

It's Logistic Regression's turn!

O |   |   |   |   |  
- + - + - + - + - + -
X | X |   |   |   |  
- + - + - + - + - + -
X | O |   |   |   |  
- + - + - + - + - + -
O | X |   |   |   |  
- + - + - + - + - + -
X | O |   |   |   | O
- + - + - + - + - + -
O | X |   |   |   | O

It's MiniMax's turn!
I have compared  2858  combinations and executed the best move out of it. I can't lose, dude!

O | X |   |   |   |  
- + - + - + - + - + -
X | X |   |   |   |  
- + - + - + - + - + -
X | O |   |   |   |  
- + - + - + - + - + -
O | X |   |   |   |  
- + - + - + - + - + -
X | O |   |   |   | O
- + - + - + - + - + -
O | X |   |   |   | O

It's Logistic Regression's turn!

O | X |   |   |   |  
- + - + - + - + - + -
X | X |   |   |   |  
- + - + - + - + - + -
X | O |   |   |   |  
- + - + - + - + - + -
O | X |   |   |   | O
- + - + - + - + - + -
X | O |   |   |   | O
- + - + - + - + - + -
O | X |   |   |   | O

It's MiniMax's turn!
I have compared  974  combinations and executed the best move out of it. I can't lose, dude!

O | X |   |   |   |  
- + - + - + - + - + -
X | X |   |   |   |  
- + - + - + - + - + -
X | O |   |   |   | X
- + - + - + - + - + -
O | X |   |   |   | O
- + - + - + - + - + -
X | O |   |   |   | O
- + - + - + - + - + -
O | X |   |   |   | O

It's Logistic Regression's turn!

O | X |   |   |   |  
- + - + - + - + - + -
X | X |   |   |   | O
- + - + - + - + - + -
X | O |   |   |   | X
- + - + - + - + - + -
O | X |   |   |   | O
- + - + - + - + - + -
X | O |   |   |   | O
- + - + - + - + - + -
O | X |   |   |   | O

It's MiniMax's turn!
I have compared  891  combinations and executed the best move out of it. I can't lose, dude!

O | X |   |   |   |  
- + - + - + - + - + -
X | X |   |   |   | O
- + - + - + - + - + -
X | O |   |   |   | X
- + - + - + - + - + -
O | X |   |   |   | O
- + - + - + - + - + -
X | O |   |   |   | O
- + - + - + - + - + -
O | X | X |   |   | O

It's Logistic Regression's turn!

O | X |   |   |   | O
- + - + - + - + - + -
X | X |   |   |   | O
- + - + - + - + - + -
X | O |   |   |   | X
- + - + - + - + - + -
O | X |   |   |   | O
- + - + - + - + - + -
X | O |   |   |   | O
- + - + - + - + - + -
O | X | X |   |   | O

It's MiniMax's turn!
I have compared  267  combinations and executed the best move out of it. I can't lose, dude!

O | X |   |   |   | O
- + - + - + - + - + -
X | X |   |   |   | O
- + - + - + - + - + -
X | O |   |   |   | X
- + - + - + - + - + -
O | X |   |   |   | O
- + - + - + - + - + -
X | O |   |   |   | O
- + - + - + - + - + -
O | X | X | X |   | O

It's Logistic Regression's turn!

O | X |   |   |   | O
- + - + - + - + - + -
X | X |   |   |   | O
- + - + - + - + - + -
X | O |   |   |   | X
- + - + - + - + - + -
O | X |   |   |   | O
- + - + - + - + - + -
X | O | O |   |   | O
- + - + - + - + - + -
O | X | X | X |   | O

It's MiniMax's turn!
I have compared  191  combinations and executed the best move out of it. I can't lose, dude!

O | X |   |   |   | O
- + - + - + - + - + -
X | X |   |   |   | O
- + - + - + - + - + -
X | O |   |   |   | X
- + - + - + - + - + -
O | X |   |   |   | O
- + - + - + - + - + -
X | O | O |   |   | O
- + - + - + - + - + -
O | X | X | X | X | O

It's over!
Minimax wins!
==============================================================
Minimax:  1 Logistic Regression:  0 Draw:  0
==============================================================
```

### Battle 4: Logistic Regression Vs ConvNet!
Well, as expected, ConvNet didn't perform well, because it's a very small board to capture necessary patterns. ConvNet performed as same as a random move generator! That doesn't mean that Logistic Regression was very impressive. It won the battle, but the moves were like kid sometimes!

Data has been gereated by placing some random moves against minimax algorithm.

Below is the final battle between Logistic Regression Vs ConvNet!. 

```python
Battle number:  4
It's ConvNet's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   | X |   |   |   |   |  

It's Logistic Regression's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
O |   | X |   |   |   |   |  

It's ConvNet's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
O |   | X | X |   |   |   |  

It's Logistic Regression's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   | O |   |   |   |  
- + - + - + - + - + - + - + -
O |   | X | X |   |   |   |  

It's ConvNet's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   | X | O |   |   |   |  
- + - + - + - + - + - + - + -
O |   | X | X |   |   |   |  

It's Logistic Regression's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   | X | O |   |   |   |  
- + - + - + - + - + - + - + -
O | O | X | X |   |   |   |  

It's ConvNet's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   | X |   |   |   |  
- + - + - + - + - + - + - + -
  |   | X | O |   |   |   |  
- + - + - + - + - + - + - + -
O | O | X | X |   |   |   |  

It's Logistic Regression's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   | X |   |   |   |  
- + - + - + - + - + - + - + -
  |   | X | O |   |   |   |  
- + - + - + - + - + - + - + -
O | O | X | X |   | O |   |  

It's ConvNet's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   | X |   |   |   |  
- + - + - + - + - + - + - + -
  |   | X | O |   |   |   |  
- + - + - + - + - + - + - + -
O | O | X | X |   | O | X |  

It's Logistic Regression's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   | X |   |   |   |  
- + - + - + - + - + - + - + -
  |   | X | O |   | O |   |  
- + - + - + - + - + - + - + -
O | O | X | X |   | O | X |  

It's ConvNet's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   | X |   |   |   |  
- + - + - + - + - + - + - + -
  |   | X | O |   | O | X |  
- + - + - + - + - + - + - + -
O | O | X | X |   | O | X |  

It's Logistic Regression's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   | X |   |   | O |  
- + - + - + - + - + - + - + -
  |   | X | O |   | O | X |  
- + - + - + - + - + - + - + -
O | O | X | X |   | O | X |  

It's ConvNet's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   | X | X |   |   | O |  
- + - + - + - + - + - + - + -
  |   | X | O |   | O | X |  
- + - + - + - + - + - + - + -
O | O | X | X |   | O | X |  

It's Logistic Regression's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   | O |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   | X | X |   |   | O |  
- + - + - + - + - + - + - + -
  |   | X | O |   | O | X |  
- + - + - + - + - + - + - + -
O | O | X | X |   | O | X |  

It's ConvNet's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   | O |   |   |   | X |  
- + - + - + - + - + - + - + -
  |   | X | X |   |   | O |  
- + - + - + - + - + - + - + -
  |   | X | O |   | O | X |  
- + - + - + - + - + - + - + -
O | O | X | X |   | O | X |  

It's Logistic Regression's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   | O |   |   |   | X |  
- + - + - + - + - + - + - + -
  |   | X | X |   |   | O |  
- + - + - + - + - + - + - + -
  |   | X | O |   | O | X |  
- + - + - + - + - + - + - + -
O | O | X | X |   | O | X | O

It's ConvNet's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   | O |   |   |   | X |  
- + - + - + - + - + - + - + -
  |   | X | X |   |   | O |  
- + - + - + - + - + - + - + -
  |   | X | O |   | O | X | X
- + - + - + - + - + - + - + -
O | O | X | X |   | O | X | O

It's Logistic Regression's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   | O |   |   |   | X |  
- + - + - + - + - + - + - + -
  |   | X | X |   |   | O | O
- + - + - + - + - + - + - + -
  |   | X | O |   | O | X | X
- + - + - + - + - + - + - + -
O | O | X | X |   | O | X | O

It's ConvNet's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   | X |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   | O |   |   |   | X |  
- + - + - + - + - + - + - + -
  |   | X | X |   |   | O | O
- + - + - + - + - + - + - + -
  |   | X | O |   | O | X | X
- + - + - + - + - + - + - + -
O | O | X | X |   | O | X | O

It's Logistic Regression's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   | X |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   | O |   |   |   | X |  
- + - + - + - + - + - + - + -
  |   | X | X |   | O | O | O
- + - + - + - + - + - + - + -
  |   | X | O |   | O | X | X
- + - + - + - + - + - + - + -
O | O | X | X |   | O | X | O

It's ConvNet's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   | X |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   | O |   |   | X | X |  
- + - + - + - + - + - + - + -
  |   | X | X |   | O | O | O
- + - + - + - + - + - + - + -
  |   | X | O |   | O | X | X
- + - + - + - + - + - + - + -
O | O | X | X |   | O | X | O

It's Logistic Regression's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   | X |   |   | O |   |  
- + - + - + - + - + - + - + -
  |   | O |   |   | X | X |  
- + - + - + - + - + - + - + -
  |   | X | X |   | O | O | O
- + - + - + - + - + - + - + -
  |   | X | O |   | O | X | X
- + - + - + - + - + - + - + -
O | O | X | X |   | O | X | O

It's ConvNet's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   | X |   |  
- + - + - + - + - + - + - + -
  |   | X |   |   | O |   |  
- + - + - + - + - + - + - + -
  |   | O |   |   | X | X |  
- + - + - + - + - + - + - + -
  |   | X | X |   | O | O | O
- + - + - + - + - + - + - + -
  |   | X | O |   | O | X | X
- + - + - + - + - + - + - + -
O | O | X | X |   | O | X | O

It's Logistic Regression's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   | X |   |  
- + - + - + - + - + - + - + -
  |   | X |   |   | O |   |  
- + - + - + - + - + - + - + -
  |   | O |   |   | X | X |  
- + - + - + - + - + - + - + -
  |   | X | X |   | O | O | O
- + - + - + - + - + - + - + -
  | O | X | O |   | O | X | X
- + - + - + - + - + - + - + -
O | O | X | X |   | O | X | O

It's ConvNet's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   | X |   |  
- + - + - + - + - + - + - + -
  |   | X |   |   | O |   |  
- + - + - + - + - + - + - + -
  |   | O |   |   | X | X |  
- + - + - + - + - + - + - + -
  |   | X | X |   | O | O | O
- + - + - + - + - + - + - + -
X | O | X | O |   | O | X | X
- + - + - + - + - + - + - + -
O | O | X | X |   | O | X | O

It's Logistic Regression's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   | X |   |  
- + - + - + - + - + - + - + -
  |   | X |   |   | O |   |  
- + - + - + - + - + - + - + -
  |   | O |   |   | X | X |  
- + - + - + - + - + - + - + -
  | O | X | X |   | O | O | O
- + - + - + - + - + - + - + -
X | O | X | O |   | O | X | X
- + - + - + - + - + - + - + -
O | O | X | X |   | O | X | O

It's ConvNet's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   | X |   |   | X |   |  
- + - + - + - + - + - + - + -
  |   | X |   |   | O |   |  
- + - + - + - + - + - + - + -
  |   | O |   |   | X | X |  
- + - + - + - + - + - + - + -
  | O | X | X |   | O | O | O
- + - + - + - + - + - + - + -
X | O | X | O |   | O | X | X
- + - + - + - + - + - + - + -
O | O | X | X |   | O | X | O

It's Logistic Regression's turn!

  |   |   |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   | O |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   | X |   |   | X |   |  
- + - + - + - + - + - + - + -
  |   | X |   |   | O |   |  
- + - + - + - + - + - + - + -
  |   | O |   |   | X | X |  
- + - + - + - + - + - + - + -
  | O | X | X |   | O | O | O
- + - + - + - + - + - + - + -
X | O | X | O |   | O | X | X
- + - + - + - + - + - + - + -
O | O | X | X |   | O | X | O

It's ConvNet's turn!

  |   | X |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   | O |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   | X |   |   | X |   |  
- + - + - + - + - + - + - + -
  |   | X |   |   | O |   |  
- + - + - + - + - + - + - + -
  |   | O |   |   | X | X |  
- + - + - + - + - + - + - + -
  | O | X | X |   | O | O | O
- + - + - + - + - + - + - + -
X | O | X | O |   | O | X | X
- + - + - + - + - + - + - + -
O | O | X | X |   | O | X | O

It's Logistic Regression's turn!

  |   | X |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   | O |   |   |   |   |  
- + - + - + - + - + - + - + -
  |   | X |   |   | X |   |  
- + - + - + - + - + - + - + -
  |   | X |   |   | O |   |  
- + - + - + - + - + - + - + -
  | O | O |   |   | X | X |  
- + - + - + - + - + - + - + -
  | O | X | X |   | O | O | O
- + - + - + - + - + - + - + -
X | O | X | O |   | O | X | X
- + - + - + - + - + - + - + -
O | O | X | X |   | O | X | O

I have learnt lots of new tricks!
Logistic Regression wins!
Battle number:  5
==============================================================
ConvNet:  1 Logistic Regression:  3 Draw:  0
==============================================================
```

### Future Work
Actually there are tons of places for improvement. 
First, need to implement alpha-beta pruning to make minimax computationally efficient.
Second, my ML models are getting overfitted. Some moves are not better than a random move! A few things need to be done here to improve ML models-
* Experiment with the hyper-parameters to find good parameters for both logistic regression and ConvNet.
* Need to come up with a better approach to generate data. Right now I am using minimax and random move to generate data, which is not good enough. Especially, it failed to generate all possible data. Thus, ML models has failed to learn properly.
* Need more research and experiment on feature selection for input data. 
* Need to generate a big chunk of data.

Just wanted to explore Minimax and ML algorithms in board games, so that's all for now! Will think about the improvement in the future.. Adiós Amigo!