import numpy as np
import torch
import utils

class Dataset(torch.utils.data.Dataset):
    #def __init__(self, Ds, objs, diversities, positions, obj_ids):
    def __init__(self, datasets, calc_median=True, use_neighbours=False):
        super(Dataset, self).__init__()
        
        self.datasets = datasets
        self.use_neighbours=use_neighbours
        
        self.total_num_rows = 0
        self.num_rows = np.zeros(len(datasets), dtype=int)
        self.max_pos = np.zeros((len(datasets), 2), dtype=int)
        #self.num_frames = None
        self.num_objs = 0
        for i in range(len(datasets)):
            Ds, objs, diversity, positions, coords, neighbours = datasets[i]
            num_objects = Ds.shape[0]
            self.num_objs += num_objects
            num_frames = Ds.shape[1]
            #if self.num_frames is None:
            #    self.num_frames = num_frames
            #assert(self.num_frames >= num_frames)
            self.num_rows[i] = num_frames*num_objects
            self.total_num_rows += self.num_rows[i]
            self.max_pos[i] = np.max(positions, axis = 0)
        
        
        nx = datasets[0][0].shape[3]
        self.hanning = utils.hanning(nx, nx//4)#, num_pixel_padding=6)
        if calc_median:
            self.median = self.calc_median()
        else:
            self.median = None
            
        
               
    def __getitem__(self, index):
        for data_index in range(len(self.datasets)):
            if self.num_rows[data_index] > index:
                break
            index -= self.num_rows[data_index]

        Ds, objs, diversity, positions, coords, neighbours = self.datasets[data_index]

        num_frames = Ds.shape[1]
        
        obj_index = index//num_frames
        frame_index = index % num_frames
        #print("index, obj_index, frame_index", index, obj_index, frame_index, Ds.shape)
        num_ch = 2
        if self.use_neighbours:
            num_ch = 18#32
        Ds_out = np.empty((num_ch, Ds.shape[3], Ds.shape[4])).astype('float32')
        Ds_out[:2] = np.array(Ds[obj_index, frame_index, :, :, :])
        #if self.use_neighbours:
        #    Ds_out[2:num_ch//2] = np.tile(np.array(Ds[obj_index, frame_index, :, :, :]), (num_ch//4-1, 1, 1))  
        

        if positions is not None:
            pos_x = positions[obj_index, 0]
            pos_y = positions[obj_index, 1]
            
            if self.use_neighbours:
                ind_out = 2#num_ch // 2
                if neighbours is not None:
                    for ind in neighbours[obj_index]:
                        # The neighbouring patch may not exist in this dataset
                        # due to the split to train and validation sets
                        if ind >= 0 and ind < Ds.shape[0]:
                            Ds_out[ind_out:ind_out+2] = np.array(Ds[ind, frame_index, :, :, :])
                        else:
                            # Fill void patches if the object was on the edge of field
                            Ds_out[ind_out:ind_out+2] = np.array(Ds[obj_index, frame_index, :, :, :])
                        ind_out += 2
                else:
                    # Just for backward compatibility
                    max_pos = self.max_pos[data_index]
                    num_neighbours = 8
                    if pos_x == 0 or pos_x == max_pos[0]:
                        num_neighbours -= 3
                    if pos_y == 0 or pos_y == max_pos[1]:
                        num_neighbours -= 3
                    if num_neighbours == 2: # corner
                        num_neighbours = 3
                    num_found = 0
                    # First look around the given object
                    for ind in np.arange(max(obj_index-num_neighbours, 0), min(obj_index+num_neighbours+1, Ds.shape[0]-1)):
                        pos = positions[ind]
                        if pos[0] >= pos_x - 1 and pos[0] <= pos_x + 1:
                            if pos[1] >= pos_y - 1 and pos[1] <= pos_y + 1:
                                if pos[0] != pos_x or pos[1] != pos_y:
                                    Ds_out[ind_out:ind_out+2] = np.array(Ds[ind, frame_index, :, :, :])
                                    ind_out += 2
                                    num_found += 1
                                    if num_found == num_neighbours:
                                        break
                    # Now make full search
                    if num_found < num_neighbours:
                        ind = 0
                        for pos in positions:
                            if pos[0] >= pos_x - 1 and pos[0] <= pos_x + 1:
                                if pos[1] >= pos_y - 1 and pos[1] <= pos_y + 1:
                                    if pos[0] != pos_x or pos[1] != pos_y:
                                        Ds_out[ind_out:ind_out+2] = np.array(Ds[ind, frame_index, :, :, :])
                                        ind_out += 2
                                        num_found += 1
                                        if num_found == num_neighbours:
                                            break
                            ind += 1
                    # Fill void patches if the object was on the edge of field
                    for ind_out in np.arange(ind_out, num_ch, step=2):
                        Ds_out[ind_out:ind_out+2] = np.array(Ds[obj_index, frame_index, :, :, :])
                    

        if diversity is not None:
            diversities_out = np.zeros((Ds.shape[2], Ds.shape[3], Ds.shape[4])).astype('float32')
            if positions is None:
                if len(diversity.shape) == 3:
                    # Diversities both for focus and defocus image
                    #diversity_out[k, :, :, 0] = diversity_in[0]
                    for div_i in np.arange(diversity.shape[0]):
                        diversities_out[1, :, :] += diversity[div_i]
                else:
                    assert(len(diversity.shape) == 2)
                    # Just a defocus
                    diversities_out[1, :, :] = diversity
            else:
                assert(len(diversity.shape) == 5)
                #for div_i in np.arange(diversity_in.shape[2]):
                #    #diversity_out[k, :, :, 0] = diversity_in[positions[i, 0], positions[i, 1], 0]
                #    diversity_out[k, :, :, 1] += diversity_in[positions[i, 0], positions[i, 1], div_i]
                #    #diversity_out[k, :, :, 1] = diversity_in[positions[i, 0], positions[i, 1], 1]
                diversities_out[1, :, :] += diversity[pos_x, pos_y, 1]
        else:
            diversities_out  = None

        #med = np.median(Ds_out, axis=(0, 1, 2), keepdims=True)
        #Ds_out -= med
        #Ds_out = self.hanning.multiply(Ds_out, axis=1)
        #Ds_out += med
        #Ds_out /= med
        
        #######################################################################
        # DEBUG
        #if index < batch_size:
        #    my_test_plot = plot.plot(nrows=Ds_out.shape[0]//2, ncols=4)
        #    for i in range(Ds_out.shape[0]//2):
        #        my_test_plot.colormap(Ds_out[2*i], [i, 0], show_colorbar=True)
        #        my_test_plot.colormap(Ds_out[2*i+1], [i, 1], show_colorbar=True)
        #        my_test_plot.colormap(diversities_out[0], [i, 2], show_colorbar=True)
        #        my_test_plot.colormap(diversities_out[1], [i, 3], show_colorbar=True)
        #    my_test_plot.save(f"{dir_name}/Ds_dbg{index}.png")
        #    my_test_plot.close()
        #######################################################################
        
        return Ds_out, diversities_out
      

    def get_data(self):
        Ds0, div0 = self[0]
        Ds = np.empty((self.length(), Ds0.shape[0], Ds0.shape[1], Ds0.shape[2]))
        diversities = np.empty((self.length(), div0.shape[0], div0.shape[1], div0.shape[2]))
        for i in range(self.length()):
            Ds_i, diversity_i = self[i]
            Ds[i] = Ds_i
            diversities[i] = diversity_i
        return Ds, diversities
        
    
    def __len__(self):
        return self.total_num_rows
    
    def length(self):
        return self.total_num_rows

    def get_num_objs(self):
        return self.num_objs

    #def get_num_frames(self):
    #    return self.num_frames

    def get_positions(self):
        # In test mode we assume currently that there is only one full image
        assert(len(self.datasets) == 1)
        return self.datasets[0][3]
    
    def get_coords(self):
        # In test mode we assume currently that there is only one full image
        assert(len(self.datasets) == 1)
        return self.datasets[0][4]
        
    def get_obj_data(self, index):
        for data_index in range(len(self.datasets)):
            if self.num_rows[data_index] > index:
                break
            index -= self.num_rows[data_index]

        Ds, objs, diversity, positions, coords, neighbours = self.datasets[data_index]

        num_frames = Ds.shape[1]
        
        obj_index = index//num_frames
        
        if objs is not None:
            obj = objs[obj_index]
        else:
            obj = None
            
        if positions is not None:
            pos = positions[obj_index]
        else:
            pos = None
            
        if coords is not None:
            coord = coords[obj_index]
        else:
            coord = None
        
        return obj_index, obj, pos, coord
    
    def calc_median(self):
        # This is actually mean of medians
        med = 0.
        for i in range(self.length()):
            Ds, _ = self[i]
            med += np.median(Ds)
        return (med/self.length()).astype("float32")
