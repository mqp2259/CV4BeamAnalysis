# Made by Isaac Joffe

import relationships    # for calling deep learning model

CONF_THRES = 0.5    # threshold for activation from deep learning model


"""
Class to represent a beam. Stores raw information about the beam and its associated supports as well as
more processed data, such as the beam's orientation and length. Stores type and position of each attached
support and load for processing.
"""
class Beam:
    def __init__(self, info):
        self.__raw_position = [info[0], info[1], info[2], info[3]]    # x_min, y_min, x_max, y_max
        self.__orientation = (info[2] - info[0]) >= (info[3] - info[1])    # True means axis of beam is x-axis, False means y-axis
        if self.get_orientation():    # horizontal beam
            self.__centre = (info[1] + info[3]) / 2    # centre coordinate of beam
        else:    # vertical beam
            self.__centre = (info[0] + info[2]) / 2    # centre coordinate of beam
        self.__nodes = []    # a list of all supports and loads
        return

    def get_raw_position(self):
        return self.__raw_position

    def get_x_min(self):
        return self.__raw_position[0]

    def get_y_min(self):
        return self.__raw_position[1]

    def get_x_max(self):
        return self.__raw_position[2]

    def get_y_max(self):
        return self.__raw_position[3]

    def get_orientation(self):
        return self.__orientation

    def get_centre(self):
        return self.__centre

    def get_min(self):
        if self.get_orientation():    # horizontal beam
            return self.get_x_min()
        else:    # vertical beam
            return self.get_y_min()

    def get_max(self):
        if self.get_orientation():    # horizontal beam
            return self.get_x_max()
        else:    # vertical beam
            return self.get_y_max()

    def get_nodes(self):
        return self.__nodes

    def add_node(self, node):
        #if isinstance(node, Load):
            #node.set_values(self)    # use beam information to deduce directions of forces
        self.__nodes.append(node)
        return


"""
Class to represents supports and loads. Only holds positional data and provides uniform access interface for supports and loads.
"""
class Node:
    def __init__(self, info):
        self.__raw_position = [info[0], info[1], info[2], info[3]]    # x_min, y_min, x_max, y_max
        return

    def get_raw_position(self):
        return self.__raw_position

    def get_x_min(self):
        return self.__raw_position[0]

    def get_y_min(self):
        return self.__raw_position[1]

    def get_x_max(self):
        return self.__raw_position[2]

    def get_y_max(self):
        return self.__raw_position[3]

    def get_min(self, beam):
        if beam.get_orientation():    # horizontal beam
            return self.get_x_min()
        else:    # vertical beam
            return self.get_y_min()

    def get_max(self, beam):
        if beam.get_orientation():    # horizontal beam
            return self.get_x_max()
        else:    # vertical beam
            return self.get_y_max()
            
    def get_centre(self, beam):
        if beam.get_orientation():    # horizontal beam
            centre = (self.get_x_max() + self.get_x_min()) / 2
            if centre < beam.get_x_min():
                centre = beam.get_x_min()    # support cannot exist before the beam if it is connected
            if centre > beam.get_x_max():
                centre = beam.get_x_max()    # support cannot exist after the beam if it is connected
        else:    # vertical beam
            centre = (self.get_y_max() + self.get_y_min()) / 2
            if centre < beam.get_y_min():
                centre = beam.get_y_min()
            if centre > beam.get_y_max():
                centre = beam.get_y_max()
        return centre


"""
Class to represent a beam support. Merely holds raw data and implements an interface for accessing this data.
"""
class Support(Node):
    def __init__(self, info):
        Node.__init__(self, info)
        self.__support_type = info[-1]    # 0 means fix, 1 means pin, 2 means roller
        return

    def get_support_type(self):
        return self.__support_type


"""
Class to represent a beam load. Merely holds raw data and implements an interface for accessing this data.
"""
class Load(Node):
    def __init__(self, info, value):
        Node.__init__(self, info)
        self.__load_type = info[-1]    # 9, 10, 11, 12 means point load, 13, 14, 15, 16 means distributed load, 7, 8 means couple
        self.__magnitude = value[-2]    # second-to-last value is where number-reader writes value
        if self.get_load_type() == 7:
            self.__x_force = 0
            self.__y_force = 0
            self.__moment = -self.get_magnitude()
        elif self.get_load_type() == 8:
            self.__x_force = 0
            self.__y_force = 0
            self.__moment = self.get_magnitude()
        elif self.get_load_type() == 9:
            self.__x_force = self.get_magnitude()
            self.__y_force = 0
            self.__moment = 0
        elif self.get_load_type() == 10:
            self.__x_force = -self.get_magnitude()
            self.__y_force = 0
            self.__moment = 0
        elif self.get_load_type() == 11:
            self.__x_force = 0
            self.__y_force = self.get_magnitude()
            self.__moment = 0
        elif self.get_load_type() == 12:
            self.__x_force = 0
            self.__y_force = -self.get_magnitude()
            self.__moment = 0
        elif self.get_load_type() == 13:
            self.__x_force = self.get_magnitude()
            self.__y_force = 0
            self.__moment = 0
        elif self.get_load_type() == 14:
            self.__x_force = -self.get_magnitude()
            self.__y_force = 0
            self.__moment = 0
        elif self.get_load_type() == 15:
            self.__x_force = 0
            self.__y_force = self.get_magnitude()
            self.__moment = 0
        elif self.get_load_type() == 16:
            self.__x_force = 0
            self.__y_force = -self.get_magnitude()
            self.__moment = 0
        return

    def get_load_type(self):
        return self.__load_type

    def get_magnitude(self):
        return self.__magnitude
    
    def get_x_force(self):
        return self.__x_force

    def get_y_force(self):
        return self.__y_force

    def get_moment(self):
        return self.__moment


"""
Function to convert a list of objects recognized by the computer vision algorithm into a useful beam representation.

    Parameters:
        feature (list of lists) : raw data on features detected by computer vision algorithm
        model (TensorFlow sequyential model) : deep learning model to use

    Returns:
        beams (list of Beam objects) : a list of all processed beams in the image
"""
def beamify(features, bs_model, bl_model, ln_model):
    # store the beams associated with an image as a list
    beams = []
    # iterate over each beam-support pair in the image and determine if they are connected
    for element1 in features:
        if element1[-1] == 3:
            beam = Beam(element1)    # another beam is discovered in the image
            for element2 in features:
                if element2[-1] in [0, 1, 2]:
                    # a new support (new to this beam) has been discovered
                    # get the coordinates of the beam and the support in the image
                    instance = []
                    for i in range(4):
                        instance.append(element1[i])
                    for i in range(4):
                        instance.append(element2[i])
                    # determine the largest and smallest coordinates for normalization
                    xmin = min(instance[0], instance[2], instance[4], instance[6])
                    xmax = max(instance[0], instance[2], instance[4], instance[6])
                    ymin = min(instance[1], instance[3], instance[5], instance[7])
                    ymax = max(instance[1], instance[3], instance[5], instance[7])
                    # convert coordinates to fractional form so smallest is 0 and largest is 1
                    for i in range(len(instance)):
                        if i % 2 == 0:
                            instance[i] = (instance[i] - xmin) / (xmax - xmin)
                        else:
                            instance[i] = (instance[i] - ymin) / (ymax - ymin)

                    # associate the support to the beam if they are determined to be connected by the model
                    if relationships.test_single_instance(bs_model, instance) > CONF_THRES:
                        beam.add_node(Support(element2))
                
                elif element2[-1] in [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
                    # a new load (new to this beam) has been discovered
                    # get the coordinates of the beam and the load in the image
                    instance = []
                    for i in range(4):
                        instance.append(element1[i])
                    for i in range(4):
                        instance.append(element2[i])
                    # determine the largest and smallest coordinates for normalization
                    xmin = min(instance[0], instance[2], instance[4], instance[6])
                    xmax = max(instance[0], instance[2], instance[4], instance[6])
                    ymin = min(instance[1], instance[3], instance[5], instance[7])
                    ymax = max(instance[1], instance[3], instance[5], instance[7])
                    # convert coordinates to fractional form so smallest is 0 and largest is 1
                    for i in range(len(instance)):
                        if i % 2 == 0:
                            instance[i] = (instance[i] - xmin) / (xmax - xmin)
                        else:
                            instance[i] = (instance[i] - ymin) / (ymax - ymin)

                    # associate the load to the beam if they are determined to be connected by the model
                    if relationships.test_single_instance(bl_model, instance) > CONF_THRES:
                        best_number = []    # data of current most likely number to be associated
                        best_number_conf = 0    # confidence that this number is the one that is associated
                        for element3 in features:
                            if element3[-1] == 4:
                                # a new number (new to this load) has been discovered
                                # get the coordinates of the load and the number in the image
                                instance = []
                                for i in range(4):
                                    instance.append(element2[i])
                                for i in range(4):
                                    instance.append(element3[i])
                                # determine the largest and smallest coordinates for normalization
                                xmin = min(instance[0], instance[2], instance[4], instance[6])
                                xmax = max(instance[0], instance[2], instance[4], instance[6])
                                ymin = min(instance[1], instance[3], instance[5], instance[7])
                                ymax = max(instance[1], instance[3], instance[5], instance[7])
                                # convert coordinates to fractional form so smallest is 0 and largest is 1
                                for i in range(len(instance)):
                                    if i % 2 == 0:
                                        instance[i] = (instance[i] - xmin) / (xmax - xmin)
                                    else:
                                        instance[i] = (instance[i] - ymin) / (ymax - ymin)

                                # if this number is more likely to be associated, update the current number
                                if relationships.test_single_instance(ln_model, instance) > best_number_conf:
                                    best_number_conf = relationships.test_single_instance(ln_model, instance)    # new confidence to beat
                                    best_number = element3

                        beam.add_node(Load(element2, best_number))

            beams.append(beam)    # completed beam is appended to the list

    return beams


def main():
    return


if __name__ == "__main__":
    main()
