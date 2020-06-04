import numpy as np

class floodfill:
    
    def __init__(self, mat, compFunc = (lambda x, y: x == y), mask = None):
        self.mat = mat
        self.label = 0
        self.closedRegions = set()
        self.labels = np.zeros_like(mat)
        self.label = 1
        self.neighbors = dict()
        self.regionAreas = dict()
        self.regionExtents = dict()
        self.compFunc = compFunc
        self.mask = mask

    def fill(self, row, col):
        if self.labels[row, col] == 0:
            if self.mask is not None:
                self.maskValue = self.mask[row, col]
            self.area = 0
            self.minRow = row
            self.maxRow = row
            self.minCol = col
            self.maxCol = col
            self.closedRegions.add(self.label)
            self.fillConnectedRegion(row, col)
            self.regionAreas[self.label] = self.area
            self.label += 1
            self.regionExtents[self.label] = [self.minRow, self.maxRow, self.minCol, self.maxCol]


    def fillConnectedRegion(self, row, col):
        if row < self.minRow:
            self.minRow = row
        if row > self.maxRow:
            self.maxRow = row
        startCol, endCol = self.fillRow(row, col)
    
        self.area += endCol - startCol + 1
    
        if startCol < self.minCol:
            self.minCol = startCol
        if endCol > self.maxCol:
            self.maxCol = endCol
    
    
        for col1 in np.arange(startCol, endCol+1):
            value = self.mat[row, col1]
            if row > 0 and self.labels[row - 1, col1] == 0:
                neighborValue = self.mat[row - 1, col1]
                if self.checkMask(row - 1, col1) and self.compFunc(neighborValue, value):
                    self.fillConnectedRegion(row - 1, col1)
                else:
                    self.updateClosedRegions(neighborValue)
            if row < self.mat.rows - 1 and self.labels[row + 1, col1] == 0:
                neighborValue = self.mat[row + 1, col1]
                if self.checkMask(row + 1, col1) and self.compFunc(neighborValue, value):
                    self.fillConnectedRegion(row + 1, col1)
                else:
                    self.updateClosedRegions(neighborValue)

    def fillRow(self, row, col):
        self.labels[row, col] = self.label
        initialValue = self.mat[row, col]
        value = initialValue
        for startCol in np.arange(col - 1, -1, -1):# col - 1; startCol >= 0; startCol--):
            neighborValue = self.mat[row, startCol]
            if self.checkMask(row, startCol) and self.compFunc(neighborValue, value):
                self.labels[row, startCol] = self.label
                value = neighborValue
            else:
                self.updateClosedRegions(neighborValue)
                break
        value = initialValue
        for endCol in np.arange(col + 1, self.mat.shape[1]):# (endCol = col + 1; endCol < mat.cols; endCol++):
            neighborValue = self.mat[row, endCol]
            if self.checkMask(row, endCol) and self.compFunc(neighborValue, value):
                self.labels[row, endCol] = self.label
                value = neighborValue
            else:
                self.updateClosedRegions(neighborValue)
                break
        return startCol + 1, endCol - 1


    def updateClosedRegions(self, neighborValue):
        if self.label not in self.neighbors.keys():
            self.neighbors[self.label] = neighborValue
        elif self.neighbors[self.label] != neighborValue:
            self.closedRegions.remove(self.label);
    
    def checkMask(self, row, col):
        return self.mask is None or self.mask[row, col] == self.maskValue
