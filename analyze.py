# Made by Isaac Joffe and Mohammad Talebi Kalaleh

from openseespy.opensees import *    # for structural analysis
import opsvis as opsv    # for visualization of analysis
import matplotlib.pyplot as plt    # for displaying diagrams
import matplotlib
from pathlib import Path    # for saving results to new directories, YOLO-style
# from utils.general import increment_path    # for saving results to new directories, YOLO-style

import beam
import relationships

CONF_THRES = 0.5    # threshold for activation from deep learning model
# matplotlib.use('tkagg')


"""
Define the constants used in model construction and analysis
"""
# all models are 2D so there are 2 dimensions and 3 degrees of freedom (x, y, rotation)
ndm = 2
ndf = 3
# constants relating to the properties of the beam
A = 2.e-3
Iz = 1.6e-5
E = 200.e9
# # scaling factors for visualizations
# sfacN, sfacT, sfacM = 1, 0.001, 0.000005


"""
Function to create a comprehensive structural model out of a beam data structure
"""
def create_model(beam_system, features, gn_golden, el_golden, ls_golden):
    model('basic', '-ndm', ndm, '-ndf', ndf)    # create a new 2d model

    # create all nodes corresponding to loads and supports acting on the beam
    all_nodes = beam_system.get_nodes()    # a list of loads and supports in the system
    node_mapping = []    # a list of lists containing all nodes with their position, support (at most 1), and load (at most 1)
    for current_node in all_nodes:
        if isinstance(current_node, beam.Load) and current_node.get_load_type() in [13, 14, 15, 16]:    # distributed load
            # assign a node for the beginning of the distributed load
            is_new = True    # assume this node is already supported
            for i in range(len(node_mapping)):
                if node_mapping[i][0] == current_node.get_min(beam_system):    # another node exists at this coordinate
                    node_mapping[i][3] = current_node.get_load_type()    # save type of load at this node
                    node_mapping[i][4] = current_node    # save exact info of the node at this coordinate
                    is_new = False    # this was not a new node
                    break
            if is_new:
                node_mapping.append([current_node.get_min(beam_system), None, None, current_node.get_load_type(), current_node])    # unsupported, save coordinate
            # assign a node for the end of the distributed load
            is_new = True    # assume this node is already supported
            for i in range(len(node_mapping)):
                if node_mapping[i][0] == current_node.get_max(beam_system):    # another node exists at this coordinate
                    node_mapping[i][3] = None    # no load exists at this node
                    node_mapping[i][4] = current_node    # store reference to distributed load which is now over
                    is_new = False    # this was not a new node
                    break
            if is_new:
                node_mapping.append([current_node.get_max(beam_system), None, None, None, current_node])    # unsupported, save coordinate

        elif isinstance(current_node, beam.Load):    # point loads, forces and moments
            # assign a node for the centre of the point load, which is where it acts
            is_new = True    # assume this node is already supported
            for i in range(len(node_mapping)):
                if node_mapping[i][0] == current_node.get_centre(beam_system):    # another node exists at this coordinate
                    node_mapping[i][3] = current_node.get_load_type()    # save type of load at this node
                    node_mapping[i][4] = current_node    # save exact info of the node at this coordinate
                    is_new = False    # this was not a new node
                    break
            if is_new:
                node_mapping.append([current_node.get_centre(beam_system), None, None, current_node.get_load_type(), current_node])    # unsupported, save coordinate

        elif isinstance(current_node, beam.Support):    # fix, pin, roller supports
            # assign a node for the centre of the support device, which is where it acts
            is_new = True    # assume this node is already loaded
            for i in range(len(node_mapping)):
                if node_mapping[i][0] == current_node.get_centre(beam_system):    # another node exists at this coordinate
                    node_mapping[i][1] = current_node.get_support_type()    # save type of support at this node
                    node_mapping[i][2] = current_node    # save exact info of the node at this coordinate
                    is_new = False    # this was not a new node
                    break
            if is_new:
                node_mapping.append([current_node.get_centre(beam_system), current_node.get_support_type(), current_node, None, None])    # unloaded, save coordinate
    
    # check that the beam is fullt represented by the existing nodes
    is_start = False    # assume the beam start node has not already been encountered
    is_end = False    # assume the beam ending node has not already been encountered
    for i in range(len(node_mapping)):
        if node_mapping[i][0] == beam_system.get_min():
            is_start = True    # a node already exists for the start of the beam
        if node_mapping[i][0] == beam_system.get_max():
            is_end = True    # a node already exists for the end of the beam
    if not is_start:
        node_mapping.append([beam_system.get_min(), None, None, None, None])    # a simple node for the edge of the beam
    if not is_end:
        node_mapping.append([beam_system.get_max(), None, None, None, None])    # a simple node for the edge of the beam

    # sort all nodes in they order they appear so that elements can be easily implemented
    node_mapping.sort(key=lambda x : x[0])

    # associate elements to lengths by reshaping and combinng nodes as needed
    for i in range(len(node_mapping) - 1, 0, -1):
        length = []    # information of associated length value
        length_conf = 0    # confidence in this value
        for element1 in features:
            if element1[-1] == 5:    # length
                # add coordinates of element and length for input to the perceptron
                instance = []
                if beam_system.get_orientation():    # horizontal beam
                    instance.append(node_mapping[i - 1][0])    # previous node
                    instance.append(beam_system.get_centre())
                    instance.append(node_mapping[i][0])    # current node
                    instance.append(beam_system.get_centre())
                else:    # vertical beam
                    instance.append(beam_system.get_centre())
                    instance.append(node_mapping[i - 1][0])    # previous node
                    instance.append(beam_system.get_centre())
                    instance.append(node_mapping[i][0])    # current node
                for j in range(4):
                    instance.append(element1[j])
                # preprocess input into the required format for the model input
                xmin = min(instance[0], instance[2], instance[4], instance[6])
                xmax = max(instance[0], instance[2], instance[4], instance[6])
                ymin = min(instance[1], instance[3], instance[5], instance[7])
                ymax = max(instance[1], instance[3], instance[5], instance[7])
                for j in range(len(instance)):
                    if j % 2 == 0:
                        instance[j] = (instance[j] - xmin) / (xmax - xmin)
                    else:
                        instance[j] = (instance[j] - ymin) / (ymax - ymin)
                # find the likelihood that this current length is associated with this element
                current_conf = relationships.test_single_instance(el_golden, instance)
                if current_conf > CONF_THRES and current_conf > length_conf:    # must be most confident so far and meet some minimum threshold for element to exist
                    length_conf = current_conf
                    length = element1

        if length == []:    # no length met the minimum confidence threshold, so likley not a real element
            if node_mapping[i][2] is not None:
                # the latter node has a support, so move this into the previous node
                node_mapping[i - 1][1] = node_mapping[i][1]
                node_mapping[i - 1][2] = node_mapping[i][2]
            if node_mapping[i][4] is not None:
                # the latter node has a load, so move this into the previous node
                node_mapping[i - 1][3] = node_mapping[i][3]
                node_mapping[i - 1][4] = node_mapping[i][4]
            del node_mapping[i]    # the latter node has no reason to exist anymore

        else:    # this element has an associated length, so it exists
            number = []    # information of associated length number value
            number_conf = 0    # confidence in this value
            for element1 in features:
                if element1[-1] == 4:    # number
                    # add coordinates of length and number for input to the perceptron
                    instance = []
                    for j in range(4):
                        instance.append(length[j])
                    for j in range(4):
                        instance.append(element1[j])
                    # preprocess input into the required format for the model input
                    xmin = min(instance[0], instance[2], instance[4], instance[6])
                    xmax = max(instance[0], instance[2], instance[4], instance[6])
                    ymin = min(instance[1], instance[3], instance[5], instance[7])
                    ymax = max(instance[1], instance[3], instance[5], instance[7])
                    for j in range(len(instance)):
                        if j % 2 == 0:
                            instance[j] = (instance[j] - xmin) / (xmax - xmin)
                        else:
                            instance[j] = (instance[j] - ymin) / (ymax - ymin)
                    # find the likelihood that this current number is associated with this length
                    if relationships.test_single_instance(gn_golden, instance) > number_conf:    # must be the most likely number to be stored
                        number_conf = relationships.test_single_instance(gn_golden, instance)    # new confidence to beat
                        number = element1

            node_mapping[i][0] = number[-2]    # magnitude of length is stored in the second-to-last location of the number from the reader

    node_mapping[0][0] = 0    # first node is set at origin by convention
    # check if lengths are drawn separately for each element, not overlapping
    instance = []
    for element1 in features:
        if element1[-1] == 5:    # length
            for i in range(4):
                instance.append(element1[i])    # add coordinates of each length in any order
    while len(instance) < 20:
        instance.append(0)    # pad with zeroes as is convention for the model
    while len(instance) > 20:
        instance.pop()    # remove extra data added
    xmin = min(instance[0], instance[2], instance[4], instance[6], instance[8], instance[10], instance[12], instance[14], instance[16], instance[18])
    xmax = max(instance[0], instance[2], instance[4], instance[6], instance[8], instance[10], instance[12], instance[14], instance[16], instance[18])
    ymin = min(instance[1], instance[3], instance[5], instance[7], instance[9], instance[11], instance[13], instance[15], instance[17], instance[19])
    ymax = max(instance[1], instance[3], instance[5], instance[7], instance[9], instance[11], instance[13], instance[15], instance[17], instance[19])
    for i in range(len(instance)):
        if i % 2 == 0:
            instance[i] = (instance[i] - xmin) / (xmax - xmin)
        else:
            instance[i] = (instance[i] - ymin) / (ymax - ymin)
    if relationships.test_single_instance(ls_golden, instance) > CONF_THRES:
        for i in range(1, len(node_mapping)):
            node_mapping[i][0] = node_mapping[i - 1][0] + node_mapping[i][0]    # account for past locations of nodes

    # flip all coordinates of beam if vertical since it has been upside down so far
    if not beam_system.get_orientation():
        for i in range(len(node_mapping)):
            node_mapping[i][0] = node_mapping[-1][0] - node_mapping[i][0]

    ############################################################################################
    # max_length = 0
    # max_load = 0
    # for i in range(len(node_mapping)):
    #     if node_mapping[i][0] > max_length:
    #         max_length = node_mapping[i][0]
    #     if node_mapping[i][4] is not None and node_mapping[i][4].get_magnitude() > max_load:
    #         max_load = node_mapping[i][4].get_magnitude()
    # sfac = (max_length / max_load) * 0.001
    # global sfacN; sfacN = sfac
    # global sfacV; sfacV = sfac
    # global sfacM; sfacM = sfac / 100
    ############################################################################################
    # print(node_mapping)

    # create nodes representing the beam in the structural model
    centre = 0    # the middle coordinate of the beam on its short axis, assumed everything acts through this coordinate
    for i in range(len(node_mapping)):
        if beam_system.get_orientation():    # horizontal beam
            node(i, node_mapping[i][0], centre)    # all nodes exist at the same y-coordinate
        else:    # vertical beam
            node(i, centre, node_mapping[i][0])    # all nodes exist at the same x-coordinate

    # provide constraints for each node of the beam
    for i in range(len(node_mapping)):
        if node_mapping[i][1] == 0:
            fix(i, 1, 1, 1)    # fix support constrain all 3 directions
        elif node_mapping[i][1] == 1:
            fix(i, 1, 1, 0)    # pin support constrain translation, not rotation
        elif node_mapping[i][1] == 2:
            if beam_system.get_orientation():    # horizontal beam
                fix(i, 0, 1, 0)    # assume it supports against axis of the beam
            else:    # vertical beam
                fix(i, 1, 0, 0)    # assume it supports against axis of the beam
        else:
            fix(i, 0, 0, 0)    # loads and ends do not constrain beam at all

    # create all the elements in the model
    geomTransf('Linear', 1)    # define the type of material in the model
    for i in range(len(node_mapping) - 1):
        element('elasticBeamColumn', i, i, i + 1, A, E, Iz, 1)    # elements simply join the nodes in increasing order

    # define all the loads acting on the model
    timeSeries('Constant', 1)    # basic time series
    pattern('Plain', 1, 1)    # basic pattern for all loads to be a part of
    for i in range(len(node_mapping)):
        if node_mapping[i][3] in [7, 8, 9, 10, 11, 12]:    # point loads act at a single node
            # forces exerted by load are determined by the underlying load information
            load(i, node_mapping[i][4].get_x_force(), node_mapping[i][4].get_y_force(), node_mapping[i][4].get_moment())

        elif node_mapping[i][3] in [13, 14, 15, 16]:    # distributed loads act on elements in the model
            j = i + 1
            # without going over the edge of the beam, note all elements subjected to the distributed load
            while (j < len(node_mapping)) and (node_mapping[i][4] != node_mapping[j][4]):
                j += 1
            for k in range (i, j):
                # apply to each relevant element, forces exerted are determined by the underlying load information
                if beam_system.get_orientation():    # horizotnal beam
                    eleLoad('-ele', k, '-type', '-beamUniform', node_mapping[i][4].get_y_force(), 0)    # apply forces over elements
                else:
                    eleLoad('-ele', k, '-type', '-beamUniform', node_mapping[i][4].get_x_force(), 0)    # apply forces over elements
    
    return node_mapping[-1][0]


"""
Function to complete the actual analysis of the structural model.
"""
def analyze_model():
    # define all basic model parameters for analysis
    constraints('Transformation')
    numberer('RCM')
    system('BandGeneral')
    test('NormDispIncr', 1.0e-6, 6, 2)
    algorithm('Linear')
    integrator('LoadControl', 1)
    analysis('Static')

    # run analysis of model
    analyze(1)
    return


"""
Function to display and save files of diagrams of analyzed models.
"""
def visualize(max_length, save_dir):
    max_length /= 10    # scale down lengths for proper plotting of values
    # save_dir = increment_path(Path('runs/analyze/exp'))     # make new directory in same fashion as the YOLO model
    # save_dir.mkdir(parents=True, exist_ok=True)

    printModel('-file', save_dir / 'info.txt')    # store basic information about the model

    # opsv.plot_model()    # basic diagram of structural model
    # plt.title('Visualization of Model Structure')
    # plt.savefig(str(save_dir / 'structure.png'))

    # opsv.plot_loads_2d()    # diagram of structural diagram with applied loads drawn on
    # plt.title('Visualization of Model Structure Under the Applied Loads')
    # plt.savefig(str(save_dir / 'loads.png'))

    # opsv.plot_defo()    # plot deformed shape of the model after loads have been applied
    # plt.title('Visualization of Model Deformation due to Applied Loads')
    # plt.savefig(str(save_dir / 'deformation.png'))

    # max_value = opsv.section_force_diagram_2d('N')    # obtain the maximum value for the last member as a guideline
    # sfacN = max_length / max_value[1] if max_value[1] != 0 else 0.001    # scale plots appropriately
    # plt.close()
    # opsv.section_force_diagram_2d('N', sfacN)   # plot axial force diagram of the model
    # plt.title('Axial Force Distribution')
    # plt.savefig(str(save_dir / 'afd.png'))

    # max_value = opsv.section_force_diagram_2d('T')    # obtain the maximum value for the last member as a guideline
    # sfacT = max_length / max_value[1] if max_value[1] != 0 else 0.001    # scale plots appropriately
    # plt.close()
    # opsv.section_force_diagram_2d('T', sfacT)   # plot shear force diagram of the model
    # plt.title('Shear Force Distribution')
    # plt.savefig(str(save_dir / 'sfd.png'))

    # max_value = opsv.section_force_diagram_2d('M')    # obtain the maximum value for the last member as a guideline
    # sfacM = max_length / max_value[1] if max_value[1] != 0 else 0.001    # scale plots appropriately
    # plt.close()
    # opsv.section_force_diagram_2d('M', sfacM)   # plot bending moment diagram of the model
    # plt.title('Bending Moment Distribution')
    # plt.savefig(str(save_dir / 'bmd.png'))

    max_value = opsv.section_force_diagram_2d('M')    # obtain the maximum value for the last member as a guideline
    sfacM = max_length / max_value[1] if max_value[1] != 0 else 0.001    # scale plots appropriately
    plt.close()
    opsv.section_force_diagram_2d('M', sfacM)   # plot bending moment diagram of the model
    plt.title('Bending Moment Distribution')
    plt.savefig(save_dir / "bmd.png")

    max_value = opsv.section_force_diagram_2d('T')    # obtain the maximum value for the last member as a guideline
    sfacT = max_length / max_value[1] if max_value[1] != 0 else 0.001    # scale plots appropriately
    plt.close()
    opsv.section_force_diagram_2d('T', sfacT)   # plot shear force diagram of the model
    plt.title('Shear Force Distribution')
    plt.savefig(save_dir / "sfd.png")

    max_value = opsv.section_force_diagram_2d('N')    # obtain the maximum value for the last member as a guideline
    sfacN = max_length / max_value[1] if max_value[1] != 0 else 0.001    # scale plots appropriately
    plt.close()
    opsv.section_force_diagram_2d('N', sfacN)   # plot axial force diagram of the model
    plt.title('Axial Force Distribution')
    plt.savefig(save_dir / "afd.png")

    opsv.plot_defo()    # plot deformed shape of the model after loads have been applied
    plt.title('Visualization of Model Deformation due to Applied Loads')
    plt.savefig(save_dir / "deformation.png")

    opsv.plot_loads_2d()    # diagram of structural diagram with applied loads drawn on
    plt.title('Visualization of Model Structure Under the Applied Loads')
    plt.savefig(save_dir / "loads.png")

    opsv.plot_model()    # basic diagram of structural model
    plt.title('Visualization of Model Structure')
    plt.savefig(save_dir / "structure.png")

    # start_x, start_y, dx, dy = (0, 0, 640, 550)
    # for i in range(6):
    #     if i % 3 == 0:
    #         x = start_x
    #         y = start_y  + (dy * (i//3) )
    #     plt.figure()
    #     mngr = plt.get_current_fig_manager()
    #     mngr.window.setGeometry(x, y, dx, dy)
    #     x += dx
    
    # plt.show()    # display GUI with diagrams
    # print(f"Plots saved to {save_dir}")
    return


"""
Function to create, analyze, and visualize a structural model of a beam data structure.
"""
def analyze_beam(beam_system, features, gn_golden, el_golden, ls_golden, save_dir):
    # clear any existing model set up
    wipe()
    # create a structural model based on the beam received as input
    max_length = create_model(beam_system, features, gn_golden, el_golden, ls_golden)
    # perform structural analysis on the model created
    analyze_model()
    # visualize results of analysis and save results to files
    visualize(max_length, save_dir)
    return


def main():
    return


if __name__ == "__main__":
    main()
