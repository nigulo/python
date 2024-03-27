#!python3
import random
import numpy as np
import copy

IGNORE_NONE = 0
IGNORE_ROW = 1
IGNORE_COL = 2
IGNORE_SECTOR = 3

def add(set, row, col):
    set.add(row*9+col)

def remove(set, row, col):
    set.remove(row*9+col)

def has(set, row, col):
    return (row*9+col) in set

def getRowAndCol(cell):
    return cell // 9, cell % 9
    
def equal(grid1, grid2):
    return np.all(grid1 == grid2)
    return True

def isFinished(grid):
    return all(grid > 0)

def emptyGrid():
    return np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]])

def fixedGrid():
    return np.array([[4, 8, 0, 1, 0, 3, 0, 5, 0], 
            [0, 0, 0, 0, 0, 0, 3, 2, 0], 
            [0, 0, 3, 0, 7, 0, 6, 1, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 5], 
            [0, 7, 0, 0, 0, 0, 0, 9, 0], 
            [0, 0, 2, 9, 0, 0, 0, 0, 4], 
            [8, 0, 0, 0, 0, 0, 5, 0, 2], 
            [0, 0, 6, 0, 0, 0, 0, 0, 0], 
            [0, 0, 7, 0, 9, 0, 8, 4, 0]])

def randomSolution():
    filledCells = []
    while len(filledCells) < 81:
        grid = emptyGrid()
        filledCells = findSolution(grid)
    return grid

def randomGrid(stopAt):
    while (True):
        print(f"Generating grid {stopAt}")
        grid = emptyGrid()
        unallowed = dict()
        filledCells = findSolution2(grid, [], stopAt, unallowed)
        if not solvable(grid, unallowed):
            print("Skipping unsolvable grid")
            continue
        #grid = fixedGrid()
        #unallowed = dict()
        #print("fixing grid")
        originalGrid = np.array(grid)
        for i in range(81 - len(filledCells)):
            if not fillAllowedAndUnique(grid, filledCells, unallowed):
                break
            if len(filledCells) == 81:
                print("simple solution")
                return originalGrid
        if not solvable(grid, unallowed):
            print("Skipping unsolvable grid")
            continue
        numFilledCells = len(filledCells)
        #print(grid)
        #exit()
        numSolutions = 0
        for i in range(100):#100 * (33 - stopAt)**2):
            gridCopy = np.array(grid)
            unallowedCopy = dict(unallowed)
            filledCells2 = list(filledCells)
            if len(findSolution(gridCopy, filledCells2, 81, unallowedCopy)) == 81:
                print("complex solution")
                if not possiblyUnique(gridCopy):
                    print("Skipping nonunique solution")
                    print(grid)
                    print(gridCopy)
                    continue
                filledCells3 = list(filledCells2[numFilledCells:])
                numFilledCells3 = len(filledCells3)
                unique = True
                while len(filledCells3) > 0 and numFilledCells3 - len(filledCells3) < 30:
                    row, col = filledCells3[-1]
                    filledCells3 = filledCells3[:-1]
                    num = gridCopy[row, col]
                    gridCopy[row, col] = 0
                    for newNum in range(1, 10):
                        if (newNum == num):
                            continue
                        if allowed(gridCopy, row, col, newNum):
                            for j in range(1):
                                gridCopy2 = np.array(gridCopy)
                                if findSolution(gridCopy2, list(filledCells3), 81, dict()) == 81:
                                    if not equal(gridCopy, gridCopy2):
                                        unique = False
                                        break
                            #    if j == 0:
                            #        unique = False
                            #unique = False
                            break
                if unique:
                    numSolutions += 1
                    if numSolutions == 1:
                        firstSolution = np.array(gridCopy)
            elif i == 0:
                break
            if numSolutions > 1:
                if not equal(firstSolution, gridCopy):
                    break
                numSolutions = 1
        print(stopAt, numSolutions)
        if numSolutions == 1:
            break
    return originalGrid

def findSolution(grid, filledCells = [], stopAt = 81, unallowed = dict()):
    for num in range(1, 10):
        allOptions = allowedSectors(grid, num, unallowed)
        nfc = 0
        for sectorOptions in allOptions:
            for sectorOption in sectorOptions:
                rowSector = sectorOption[0]
                colSector = sectorOption[1]
                options = sectorOption[2]
                optionIndex = random.randint(0, len(options) - 1)
                for i in list(range(optionIndex, len(options))) + list(range(optionIndex)):
                    row = rowSector*3 + options[i][0];
                    col = colSector*3 + options[i][1];
                    if allowed(grid, row, col, num, IGNORE_NONE, unallowed):
                        grid[row, col] = num;
                        filledCells.append((row, col))
                        updateUnallowed(grid, unallowed)
                        if len(filledCells) >= stopAt or not solvable(grid, unallowed) or not possiblyUnique(grid):
                            return filledCells
                        nfc += 1
                        break
            if nfc >= 9:
                break
        if len(filledCells) < num * 9:
            break
    assert(len(filledCells) < 82)
    return filledCells

def findSolution2(grid, filledCells = [], stopAt = 81, unallowed = dict()):
    nums = np.arange(1, 10)
    for index in range(9):
        np.random.shuffle(nums)
        for num in nums:
            allOptions = allowedSectors(grid, num, unallowed)
            indices = np.arange(len(allOptions))
            np.random.shuffle(indices)
            for allOptionsIndex in indices:
                sectorOptions = allOptions[allOptionsIndex]
                if len(sectorOptions) ==  0:
                    continue
                sectorOption = sectorOptions[0]
                rowSector = sectorOption[0]
                colSector = sectorOption[1]
                options = sectorOption[2]
                i = random.randint(0, len(options) - 1)
                row = rowSector*3 + options[i][0];
                col = colSector*3 + options[i][1];
                if allowed(grid, row, col, num, IGNORE_NONE, unallowed):
                    grid[row, col] = num;
                    if not solvable(grid, unallowed) or not possiblyUnique(grid):
                        grid[row, col] = 0
                        continue
                    filledCells.append((row, col))
                    updateUnallowed(grid, unallowed)
                    if len(filledCells) >= stopAt:
                        return filledCells
                    break
    assert(len(filledCells) < 82)
    return filledCells

def possiblyUnique(grid):
    for rowSector in range(3):
        for colSector in range(3):
            for rowInSector in range(3):
                row = rowSector*3 + rowInSector
                num1 = grid[row, colSector*3]
                num2 = grid[row, colSector*3 + 1]
                num3 = grid[row, colSector*3 + 2]
                if hasPermutation(grid, num1, num2, num3, (rowSector + 1) % 3, colSector, True) \
                        or hasPermutation(grid, num1, num2, num3, (rowSector + 2) % 3, colSector, True):
                    return False
            for colInSector in range(3):
                col = colSector*3 + colInSector
                num1 = grid[rowInSector*3, col]
                num2 = grid[rowInSector*3 + 1, col]
                num3 = grid[rowInSector*3 + 2, col]
                if hasPermutation(grid, num1, num2, num3, rowSector, (colSector + 1) % 3, False) \
                        or hasPermutation(grid, num1, num2, num3, rowSector, (colSector + 2) % 3, False):
                    return False
    return True                            

                    
def hasPermutation(grid, num1, num2, num3, rowSector, colSector, horizontalOrVertical):
    permutations = [[num2, num3, num1],
                    [num3, num1, num2],
                    [num2, num1, -1],
                    [num3, -1, num1],
                    [-1, num3, num2]]
    for index1 in [0, 1, 2]:
        for permutation in permutations:
            numMatched = 0
            for index2 in [0, 1, 2]:
                if permutation[index2] == 0:
                    break
                elif permutation[index2] == -1:
                    numMatched += 1
                elif horizontalOrVertical and  permutation[index2] == grid[rowSector*3 + index1, colSector*3 + index2]:
                    numMatched += 1
                elif not horizontalOrVertical and permutation[index2] == grid[rowSector*3 + index2, colSector*3 + index1]:
                    numMatched += 1
            if (numMatched == 3):
                return True
    return False
    
def solvable(grid, unallowed = dict()):
    for num in range(1, 10):
        if np.sum(grid == num) < 9:
            isAllowed = False
            for row in range(9):
                for col in range(9):
                    if allowed(grid, row, col, num, IGNORE_NONE, unallowed):
                        isAllowed = True
                        break
                if isAllowed:
                    break
            if not isAllowed:
                return False
    return True

def updateUnallowed(grid, unallowed):
    for num in range(1, 10):
        for rowSector in range(3):
            for colSector in range(3):
                options = np.asarray(optionsInSector(grid, rowSector, colSector, num, unallowed))
                if len(options) == 0:
                    continue
                for row in range(3):
                    if np.all(options[:, 0] == row):
                        for col in range(9):
                            if col not in options[:, 1] + colSector*3:
                                if num not in unallowed:
                                    unallowed[num] = set()
                                unallowed[num].add((rowSector*3 + row, col))
                for col in range(3):
                    if np.all(options[:, 1] == col):
                        for row in range(9):
                            if row not in options[:, 0] + rowSector*3:
                                if num not in unallowed:
                                    unallowed[num] = set()
                                unallowed[num].add((row, colSector*3 + col))


def fillAllowedAndUnique(grid, filledCells = [], unallowed = dict()):
    for num in range(1, 10):
        for row in range(9):
            for col in range(9):
                if allowedAndUnique(grid, row, col, num, unallowed):
                    grid[row, col] = num;
                    if not solvable(grid, unallowed):
                        return False
                    filledCells.append((row, col))
                    updateUnallowed(grid, unallowed)
    return True

def allowedSectors(grid, num, unallowed = dict()):
    allOptions = [[], [], [], [], [], [], [], [], []]
    rowSectors = np.array([0, 1, 2])
    colSectors = np.array([0, 1, 2])
    np.random.shuffle(rowSectors)
    np.random.shuffle(colSectors)
    for rowSector in rowSectors:
        for colSector in colSectors:
            options = optionsInSector(grid, rowSector, colSector, num, unallowed)
            if len(options) > 0:
                allOptions[len(options) - 1].append([rowSector, colSector, options])
    return allOptions;
    
def optionsInSector(grid, rowSector, colSector, num, unallowed = dict()):
    options = []
    rowStart = rowSector * 3
    colStart = colSector * 3
    rowEnd = rowStart + 3
    colEnd = colStart + 3
    for i in range(rowStart, rowEnd):
        for j in range(colStart, colEnd):
            if allowed(grid, i, j, num, IGNORE_NONE, unallowed):
                grid[i, j] = num
                if solvable(grid, unallowed) and possiblyUnique(grid):
                    options.append([i - rowStart, j - colStart])
                grid[i, j] = 0
    return options;

        
def allowed(grid, row, col, num, ignore = IGNORE_NONE, unallowed = dict()):
    if num == col + 1 and row > 0:
        return False
    if grid[row, col] != 0 or (len(unallowed) > 0 and num in unallowed.keys() and (row, col) in unallowed[num]):
        return False
        
    if ignore != IGNORE_COL:
        if num in grid[:, col]:
            return False

    if ignore != IGNORE_ROW:
        if num in grid[row, :]:
            return False

    if ignore != IGNORE_SECTOR:
        startRow = row - row % 3
        startCol = col - col % 3
        endRow = startRow + 3
        endCol = startCol + 3
        if num in grid[np.arange(startRow, endRow), :][:, np.arange(startCol, endCol)]:
            return False
    return True

def allowedAndUnique(grid, row, col, num, unallowed = dict()):
    if not allowed(grid, row, col, num, IGNORE_NONE, unallowed):
        return False

    unique = True
    for i in range(9):
        if i == row:
            continue
        if grid[i, col] == num or allowed(grid, i, col, num, IGNORE_COL, unallowed):
            unique = False
            break 
    if unique:
        return True
    
    unique = True
    for i in range(9):
        if i == col:
            continue
        if grid[row, i] == num or allowed(grid, row, i, num, IGNORE_ROW, unallowed):
            unique = False
            break 
    if unique:
        return True

    startRow = row - row % 3
    startCol = col - col % 3
    endRow = startRow + 3
    endCol = startCol + 3
    for i in range(startRow, endRow):
        for j in range(startCol, endCol):
            if (i, j) == (row, col):
                continue
            if grid[i, j] == num or allowed(grid, i, j, num, IGNORE_SECTOR, unallowed):
                return False
    return True

def allowedAt(grid, emptyCells, num):
    for emptyCell in emptyCells:
        row, col = getRowAndCol(emptyCell)
        if allowed(grid, row, col, num):
            return True, row, col
    return False, 0, 0

def allowedAt2(grid, emptyCells, num):
    retVal = []
    for emptyCell in emptyCells:
        row, col = getRowAndCol(emptyCell)
        if allowed(grid, row, col, num):
            retVal.append([row, col])
    return retVal

def generate():
    minNumEmptyCells = 0
    bestGrid = emptyGrid()
    for i in range(1):
        grid = randomSolution()
        solution = np.array(grid)
        indices = np.arange(0, 81)
        for j in range(1000):
            grid = np.array(solution)
            np.random.shuffle(indices)
            numEmptyCells = 0;
            emptyCells = set()
            removedNums = set()
            triedCells = set()
            check = False
            while len(triedCells) + numEmptyCells < 81:
                lastNum = 0
                num, emptyRow, emptyCol = tryEmptyCell(grid, indices, emptyCells, check, triedCells, removedNums, solution)
                if num > 0:
                    add(emptyCells, emptyRow, emptyCol)
                    removedNums.add(num)
                    numEmptyCells += 1
                if numEmptyCells > minNumEmptyCells:
                    minNumEmptyCells = numEmptyCells
                    bestGrid = np.array(grid)
    return bestGrid, minNumEmptyCells

def tryEmptyCell(grid, indices, emptyCells, check, triedCells, removedNums, solution):
    for i in indices:
        row = i // 9
        col = i % 9
        if grid[row, col] > 0 and not has(triedCells, row, col):
            if grid[row, col] > 0:
                num = grid[row, col]
                grid[row, col] = 0
                unique = True
                for newNum in range(1, 10):
                    if newNum != num:
                        if allowed(grid, row, col, newNum):
                            #if check and checkOtherOptions(grid, emptyCells, removedNums, num, 0):
                            for j in range(1):
                                newGrid = np.array(grid)
                                if check and findSolution(newGrid, 80 - len(emptyCells)) == 81:
                                    if not equal(solution, newGrid):
                                        unique = False
                                        break
                                if j == 0:
                                    unique = False
                if unique:
                    return num, row, col
                add(triedCells, row, col)
                grid[row, col] = num
    return 0, 0, 0, 0

def checkOtherOptions(grid, emptyCells, removedNums, num, recursion):
    if recursion > 3:
        return True
    if len(emptyCells) == 0 or len(removedNums) == 0:
        return True
    otherOptions = allowedAt2(grid, emptyCells, num)
    if len(otherOptions) > 0:
        containsNum = False
        if num in removedNums:
            containsNum = True
            removedNums.remove(num)
        for otherRow, otherCol in otherOptions:
            remove(emptyCells, otherRow, otherCol)
            for otherNum in removedNums:
                if checkOtherOptions(grid, emptyCells, removedNums, num, recursion + 1):
                    add(emptyCells, otherRow, otherCol)
                    if containsNum:
                        removedNums.add(num)
                    return True
            add(emptyCells, otherRow, otherCol)
        if containsNum:
            removedNums.add(num)
    return False

while True:
    stopAt = random.randint(22, 32)
    grid = randomGrid(stopAt)
    with open(f"grid{32 - stopAt}.txt", "a") as f:
        np.savetxt(f, np.asarray(grid, dtype=int), fmt="%i")
        f.write("\n")

#while (True):
#    grid, numEptyCells = generate()
#    print(numEptyCells, grid)
#    if numEptyCells > 50:
#        print(numEptyCells, grid)
#        fileName = "easy.txt"
#        if numEptyCells > 54:
#            fileName = "hard.txt"
#        elif numEptyCells > 56:
#            fileName = "expert.txt"
#        with open(fileName, "a") as f:
#            np.savetxt(f, np.asarray(grid, dtype=int), fmt="%i")
#            f.write("\n")
