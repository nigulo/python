import numpy as np

class floodfill:
    
    def __init__(self, mat, comp_func = (lambda x, y: x == y), mask = None):
        self.mat = mat
        self.label = 0
        self.closed_regions = set()
        self.labels = np.zeros_like(mat, dtype=int)
        self.label = 1
        self.neighbors = dict()
        self.region_areas = dict()
        self.region_extents = dict()
        self.comp_func = comp_func
        self.mask = mask

    def set_comp_func(self, comp_func):
        self.comp_func = comp_func

    def fill_all(self):
        for row in range(self.mat.shape[0]):
            for col in range(self.mat.shape[1]):
                self.fill(row, col)

    def fill(self, row, col):
        if self.labels[row, col]:
            return
        if self.mask is not None:
            self.mask_value = self.mask[row, col]
        self.area = 0
        self.min_row = row
        self.max_row = row
        self.min_col = col
        self.max_col = col
        self.closed_regions.add(self.label)
        self._fill_connected_region(row, col)
        self.region_areas[self.label] = self.area
        self.region_extents[self.label] = [self.min_row, self.max_row, self.min_col, self.max_col]
        self.label += 1


    def _fill_connected_region(self, row, col):
        if row < self.min_row:
            self.min_row = row
        if row > self.max_row:
            self.max_row = row
        start_col, end_col = self._fill_row(row, col)
    
        self.area += end_col - start_col + 1
    
        if start_col < self.min_col:
            self.min_col = start_col
        if end_col > self.max_col:
            self.max_col = end_col
    
        for col1 in np.arange(start_col, end_col+1):
            value = self.mat[row, col1]
            if row > 0 and self.labels[row - 1, col1] == 0:
                neighbor_value = self.mat[row - 1, col1]
                if self._check_mask(row - 1, col1) and self.comp_func(neighbor_value, value):
                    self._fill_connected_region(row - 1, col1)
                else:
                    self._update_closed_regions(neighbor_value)
            if row < self.mat.shape[0] - 1 and self.labels[row + 1, col1] == 0:
                neighbor_value = self.mat[row + 1, col1]
                if self._check_mask(row + 1, col1) and self.comp_func(neighbor_value, value):
                    self._fill_connected_region(row + 1, col1)
                else:
                    self._update_closed_regions(neighbor_value)

    def _fill_row(self, row, col):
        self.labels[row, col] = self.label
        initial_value = self.mat[row, col]
        value = initial_value
        start_col = col - 1
        for start_col in np.arange(col - 1, -1, -1):# col - 1; start_col >= 0; start_col--):
            neighbor_value = self.mat[row, start_col]
            if self._check_mask(row, start_col) and self.comp_func(neighbor_value, value):
                self.labels[row, start_col] = self.label
                value = neighbor_value
            else:
                self._update_closed_regions(neighbor_value)
                break
        value = initial_value
        end_col = col + 1
        for end_col in np.arange(col + 1, self.mat.shape[1]):# (end_col = col + 1; end_col < mat.cols; end_col++):
            neighbor_value = self.mat[row, end_col]
            if self._check_mask(row, end_col) and self.comp_func(neighbor_value, value):
                self.labels[row, end_col] = self.label
                value = neighbor_value
            else:
                self._update_closed_regions(neighbor_value)
                break
        return start_col + 1, end_col - 1


    def _update_closed_regions(self, neighbor_value):
        if self.label not in self.neighbors.keys():
            self.neighbors[self.label] = neighbor_value
        elif self.neighbors[self.label] != neighbor_value:
            self.closed_regions.discard(self.label);
    
    def _check_mask(self, row, col):
        return self.mask is None or self.mask[row, col] == self.mask_value
