"""
Code written by Hugo Ramambason (hugo_ramambason@g.harvard.edu)
Purpose of the code was to quickly prototype this new network architecture and try out a lot of different things
Code is far from optimized!
"""

import numpy as np
import time
from sklearn.utils import shuffle
from scipy.sparse import csr_matrix, identity

class network_weights(object):
    """
    Class used to initialize structure and weights of a single layer in a network
    Class takes in arguments for neurons per site (NpS), stride, feedforward receptive field radius (distance_parameter), lateral inhibition radius (lateral_distance)
    Also takes in arguments for number of NpS in previous layer (important if input has multiple channels or in deep network where layers have NpS>1)
    Also takes in argument to do with the input dimensions of the images
    For cifar10 (3-channels, (32x32x3)), first layer will take arguments input_dim = 32 and previous_nps = 3
    When building a deep network, this class will be called multiple times for each layer to initialize structured connections and weights
    Calculate output dimension of our layer (output_dim) as the input_dim/stride (i.e. input of 28x28 image with stride 2 will return output_dim=14)
    """
    def __init__(self, NpS, previous_NpS, distance_parameter, input_dim, stride, lateral_distance):
        self.distance_parameter = distance_parameter
        self.input_dim = input_dim
        self.stride = stride
        self.NpS = NpS
        self.output_dim = int(self.input_dim/self.stride)
        self.W = None
        self.L = None
        self.W_structure = None
        self.L_structure = None
        self.previous_NpS = previous_NpS
        self.lateral_distance = lateral_distance
        
    def create_h_distances(self):
        """
        Method creates structure matrix of feedforward connections (parameter c in dual variables in eqn 14)
        Connections computed for a single channel (i.e. NpS=1)
        Method takes in input_dim, stride and distance_parameter
        
        A neuron will be connected to every neuron in previous layer within a euclidean distance <= distance_parameter
        Euclidean distances > distance_parameter are set to 0
        Euclidean distances <= distance_parameter are set to 1
        Output matrix is a sparse matrix representing feedforward connections within a single channel in our network

        Matrix has dimensions MxN where M is the number of neurons in a single channel in current layer and N is number of neurons in single 
        channel in previous layer i.e. Matrix[i,j] is connection between neuron i in current layer&channel and neuron j in previous layer&channel
        """
        distances = np.zeros((self.output_dim**2, self.input_dim, self.input_dim))
        dict_input_2_position = {}
        for row_index in range(self.input_dim):
            for column_index in range(self.input_dim):
                input_index = row_index*self.input_dim + column_index
                dict_input_2_position[row_index, column_index] = input_index
                
        centers = []
        dict_output_2_position = {}
        for i in range(self.output_dim):
            for j in range(self.output_dim):
                stride_padding = self.stride/2
                neuron_center = np.array([i*self.stride + stride_padding, j*self.stride + stride_padding])
                centers.append(neuron_center)
                neuron_index = i*self.output_dim + j
                dict_output_2_position[neuron_index] = neuron_center
                for k in range(self.input_dim):
                    for l in range(self.input_dim):
                        distances[neuron_index, k,l] = np.linalg.norm(np.array([k+0.5,l+0.5])-neuron_center)
        above_threshold = distances > self.distance_parameter
        below_threshold = distances <= self.distance_parameter
        distances[above_threshold] = 0
        distances[below_threshold] = 1
        distances = distances.reshape((self.output_dim**2, self.input_dim**2))
        return distances
    
    def create_ah_distances(self):
        """
        Method creates structure matrix of lateral inhibitory connections (parameter c in dual variables in eqn 14)
        Connections computed for a single channel (i.e. NpS=1)
        Method takes in input_dim, stride and lateral_distance
        
        A neuron will be connected to every other neuron in the same layer within a euclidean distance <= lateral_distance
        Euclidean distances > lateral_distance are set to 0
        Euclidean distances <= lateral_distance are set to 1
        Output matrix is a sparse matrix representing lateral connections within a single channel in our layer

        Matrix has dimensions MxM where M is the number of neurons in a single channel in current layer
        i.e. Matrix[i,j] is connection between neuron i in current layer&channel and neuron j in current layer&channel
        """
        centers = []
        dict_output_2_position = {}
        for i in range(self.output_dim):
            for j in range(self.output_dim):
                stride_padding = self.stride/2
                neuron_center = np.array([i*self.stride + stride_padding, j*self.stride + stride_padding])
                centers.append(neuron_center)
                neuron_index = i*self.output_dim + j
                dict_output_2_position[neuron_index] = neuron_center
        distances_ah = np.zeros((self.output_dim**2, self.output_dim**2))
        for row_index in list(dict_output_2_position.keys()):
            center = dict_output_2_position[row_index]
            for column_index in list(dict_output_2_position.keys()):
                other_center = dict_output_2_position[column_index]
                distances_ah[row_index, column_index] = np.linalg.norm(other_center - center)
        above_threshold = distances_ah > self.lateral_distance
        below_threshold = distances_ah <= self.lateral_distance
        distances_ah[above_threshold] = 0
        distances_ah[below_threshold] = 1
        return distances_ah
    
    def create_L(self):
        """
        Method creates binary inhibitory structure matrix for a single layer (with multiple channels)
        This inhibitory structure matrix is computed by by stacking matrices created previously in create_ah_distance()
        This takes into account the fact that layers in our network can have multiple NpS or channels
        
        Example with single layer network with NpS = 2:
            Each neuron has inhibitory connection to two neurons within all the sites within radius < lateral_distance
            We have NpS*(output_dim^2) total neurons in our layer
            Hence our matrix will have dimensions (NpS*output_dim) x (NpS*output_dim)
        """
        mat = self.create_ah_distances()
        blocks = [[mat]*self.NpS]*self.NpS
        L_mat = np.block(blocks)
        return L_mat
    
    def create_W(self):
        """
        Method creates binary feedforward structure matrix for layer
        Our inhibitory structure matrix is built from the hebbian distance matrix previously computed in create_h_distance()
        This takes into account the fact that layers in our network can have multiple NpS

        Example with single layer network with NpS = 2:
            Each neuron has feedforward connection to every neuron/pixel within its receptive field
            Because we have NpS = 2, we need to stack sparse matrix created in create_h_distance() twice over
            Hence our matrix will have dimensions (NpS*output_dim) x (NpS_Previous_Layer*input_dim)
        """
        mat = self.create_h_distances()
        blocks = [[mat]*self.previous_NpS]*self.NpS
        W_mat = np.block(blocks)
        return W_mat
    
    def create_weights_matrix(self):
        """
        Method initializes our network
        Create structure constants for feedforward and inhibition matrices
        Initialize inhibition matrix (L) as an identity matrix
        Initialize feedfoward matrix (W) with elements drawn from normal distribution (mean:0, std:1) divided by scaling factor
        Scaling factor is the sqrt of number of pixels/neurons in previous layer a neuron in current layer is connected to
        Our initial matrices are multiplied by structure constraints (introduces sparsity through our structure)
        Hence our feedforward matrix has rows with norm of order 1
        """
        self.W_structure = self.create_W()
        self.L_structure = self.create_L()
        factor = np.sqrt(((np.sum(self.W_structure)/self.NpS)/self.output_dim**2))
        self.W = self.W_structure*np.random.normal(0, 1, (self.W_structure.shape))/factor
        self.L = self.L_structure*np.identity(self.NpS * self.output_dim**2)

class deep_network(object):
    """
    Class can be used to build single layer/deep network structure
    Class inherits from the network_weights class we use to initialize weights and structure of our individual layers in our network
    Network instantiated in block matrix structure, single sparse matrix describes multi-layer structure
    We exploit sparsity of our matrices, using scipy.sparse.csr_matrix() to convert numpy matrices to sparse format
    Pre-define number of layers and properties for each layer (feedforward radius, lateral inhibition radius, NpS, stride, etc.)

    Individual arguments passed to class as a lists, where each item in list defines value of that argument for a particular layer.
    
    EXAMPLE:
        A 2 layer network:
        - feedforward radius of 2 on the first layer and 4 on the second layer
        - lateral inhibition radius of 0 on the first layer and 4 on the second layer
        - stride of 2 between the image and the first layer and then 1 between the first layer and the second layer
        - NpS of 4 on the first layer and 16 on the second layer
        - network trained for MNIST, images have dimension 28x28 and single channel
        - activation function = tanh(gx), hence define g for each layer with parameter tanh_factors (i.e. 1 for each layer)
        - scaling factor we set to 1 for the first layer and 2 for the second layer
        - the initial euler step we use for our neural dynamics we set at 0.2
        - lr we start off at 1e-2 with a decay factor of 2 and a floor at 1e-4
        - no feedback, so gamma=0

        We create our network:

        network = deep_network(image_dim=28, channels=1, strides=[2,1], distances=[2,4], layers=2, 
                               gamma=0, lr=1e-2, lr_floor=1e-4, decay=2, distance_lateral=[0,4], 
                               tanh_factors=[1,1], mult_factors=[1,2], euler_step=0.2)
    """
    def __init__(self, image_dim, channels, NpSs, strides, distances, 
                 layers, gamma, lr, lr_floor, decay, distances_lateral, tanh_factors, mult_factors, euler_step):
        self.image_dim = image_dim
        self.channels = channels
        self.NpSs = NpSs
        self.strides = strides
        self.distances = distances
        self.lateral_distances = distances_lateral
        self.layers = layers
        self.gamma = gamma
        self.lr = lr
        self.lr_floor = lr_floor
        self.current_lr = None
        self.decay = decay
        self.conversion_tickers = []
        self.costs = []
        self.epoch=0
        self.deep_matrix_weights = None
        self.deep_matrix_structure = None
        self.deep_matrix_identity = None
        self.weights_adjustment_matrix = None
        self.weights_update_matrix = None
        self.grad_matrix = None
        self.n_images = None
        self.dict_weights = {}
        self.dimensions = []
        self.g_vec = None
        self.euler_step = euler_step
        self.tanh_factors = tanh_factors
        self.mult_factors = mult_factors
        deep_network.create_deep_network(self)
        deep_network.create_g_vec(self)
    
    def create_deep_network(self):
        """
        Method creates block structure for deep-network

        EXAMPLE:
            A 2 layer network:
            -Neural dynamics for whole system can be calculated at once with:
                      | 0        0        0    |
            W_tilda = | W1    -(L1-I)   γW2.T  |
                      | 0        W2     (L2-I) |

            Where W_tilda is a block matrix where each individual block refers to a weight matrix associated with a layer in the network
            i.e. W1 = feedforward weight matrix for the first layer in the network

            Hence:
                     |u0|          | x|
            du/dT = -|u1| + W_tilda|r1|
                     |u2|          |r2|

            Where x is the particular image being presented to the network
            and u1/u2 and r1/r2 are the corresponding outputs for each layer in the network
            these neural dynamics are run to convergence, with our convergence criteria being that ||du||/||u|| < tol for every layer in the network

        Because of the need to carry out alternating neural dynamics and weight update step (eqn. 17&19), we store multiple matrices

        Creating W_tilda (neural dynamics, eqn.17):

                                        | 0        0        0    |
            Weights matrix of format:   | W1       L1      W2.T  |           
                                        | 0        W2       L2   |

                                        | 0        0        0    |
            Structure matrix of format: | 1        -1       γ    |           
                                        | 0         1       -1   |

                                        | 0        0        0    |
            Identity matrix of format:  | 0        I        0    |           
                                        | 0        0        I    |

            W_tilda can be calculated as the element wise operation (weights*structure) + identity

        Weight update step (eqn.19):

                                              | 0        0        0    |
            Weight adjustment matrix (wam):   | 1       1/1+γ     1    |           
                                              | 0        1        1    |

                                              | 0        0        0    |
            Weight update matrix (wum):       | 1        1/2      1    |           
                                              | 0         1       1/2  |

            Each time our neural dynamics reach convergence, our update step is defined as wum*(r@r.T - wam*weights)
        """
        #create a list of network dimensions, where each entry corresponds to length of that layers u&r vectors (eqn.17)
        for i in range(self.layers+1):
            dim = int(np.prod(self.strides[:i]))
            self.dimensions.append(int((self.image_dim/dim)**2)*([self.channels]+self.NpSs)[i])
        
        #create a dict where each layer (weights, structure, etc.) are saved as an entry
        #each layer is instantiated by calling create_weights_matrix() method inherited from from network_weights
        for i in range(self.layers):
            layer_input_dim = int(self.image_dim/np.prod(self.strides[:i]))
            self.dict_weights[i]=network_weights(NpS=([self.channels]+self.NpSs)[i+1], distance_parameter=self.distances[i], 
                                                input_dim=layer_input_dim,
                                                stride = self.strides[i], previous_NpS = ([self.channels]+self.NpSs)[i], lateral_distance=self.lateral_distances[i])
            self.dict_weights[i].create_weights_matrix()
        
        matrix_block = []
        structure_block = []
        matrix_identity = []
        weight_adjustment_block = []
        gradient_update_block = []

        for i, ele_row in enumerate(self.dimensions):
            row_block = []
            struc_block = []
            row_identity_block = []
            weights_adj_block = []
            grad_update_block = []

            #define start_block and end_block to find in which sub-part of our block matrix we are
            start_block = max(i-1, 0)
            end_block = max(len(self.dimensions)-start_block-3, 0)

            #top 'row' of block matrix is all zeros, corresponds to 'zero-th' layer of network (input image)
            if i == 0:
                row_block.append(np.zeros((ele_row, np.sum(self.dimensions))))
                struc_block.append(np.zeros((ele_row, np.sum(self.dimensions))))
                row_identity_block.append(np.zeros((ele_row, np.sum(self.dimensions))))
                weights_adj_block.append(np.zeros((ele_row, np.sum(self.dimensions))))
                grad_update_block.append(np.zeros((ele_row, np.sum(self.dimensions))))

            #for every layer except the final one, we have one feedforward weights matrix, lateral inhibition and feedback from next layer
            elif i < len(self.dimensions)-1:
                if start_block > 0:
                    row_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    struc_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    row_identity_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    weights_adj_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    grad_update_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))

                row_block.append(self.dict_weights[i-1].W)
                row_block.append(self.dict_weights[i-1].L)
                row_block.append(self.dict_weights[i].W.T)
                
                struc_block.append(self.dict_weights[i-1].W_structure/self.mult_factors[i-1])
                struc_block.append(-self.dict_weights[i-1].L_structure)
                struc_block.append(self.gamma*self.mult_factors[i]*self.dict_weights[i].W_structure.T)
                
                row_identity_block.append(np.zeros((self.dict_weights[i-1].W_structure.shape)))
                row_identity_block.append(np.identity(self.dict_weights[i-1].L_structure.shape[0]))
                row_identity_block.append(np.zeros((self.dict_weights[i].W_structure.T.shape)))

                weights_adj_block.append(self.dict_weights[i-1].W_structure)
                weights_adj_block.append(self.dict_weights[i-1].L_structure/(1+self.gamma))
                weights_adj_block.append(self.dict_weights[i].W_structure.T)

                grad_update_block.append(self.dict_weights[i-1].W_structure)
                grad_update_block.append(self.dict_weights[i-1].L_structure/2)
                grad_update_block.append(self.dict_weights[i].W_structure.T)

                if end_block > 0:
                    row_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[-end_block:])))))
                    struc_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[-end_block:])))))
                    row_identity_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[-end_block:])))))
                    weights_adj_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[-end_block:])))))
                    grad_update_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[-end_block:])))))

            #for the final layer we have only feedforward weights matrix and lateral inhibition
            elif i+1 == len(self.dimensions):
                if start_block > 0:
                    row_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    struc_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    row_identity_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    weights_adj_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    grad_update_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))

                row_block.append(self.dict_weights[i-1].W)
                row_block.append(self.dict_weights[i-1].L)
                
                struc_block.append(self.dict_weights[i-1].W_structure/self.mult_factors[i-1])
                struc_block.append(-self.dict_weights[i-1].L_structure)
                
                row_identity_block.append(np.zeros((self.dict_weights[i-1].W_structure.shape)))
                row_identity_block.append(np.identity(self.dict_weights[i-1].L_structure.shape[0]))
                
                weights_adj_block.append(self.dict_weights[i-1].W_structure)
                weights_adj_block.append(self.dict_weights[i-1].L_structure)
                
                grad_update_block.append(self.dict_weights[i-1].W_structure)
                grad_update_block.append(self.dict_weights[i-1].L_structure/2)

            matrix_block.append(row_block)
            structure_block.append(struc_block)
            matrix_identity.append(row_identity_block)
            weight_adjustment_block.append(weights_adj_block)
            gradient_update_block.append(grad_update_block)

        #exploit sparsity of matrices using scipy.sparse.csr_matrix()
        self.deep_matrix_weights = csr_matrix(np.block(matrix_block))
        self.deep_matrix_structure = csr_matrix(np.block(structure_block))
        self.deep_matrix_identity = csr_matrix(np.block(matrix_identity))
        self.weights_adjustment_matrix = csr_matrix(np.block(weight_adjustment_block))
        self.weights_update_matrix = csr_matrix(np.block(gradient_update_block))
    
    def create_g_vec(self):
        """
        activation function takes the form r = tanh(gu)
        create a vector of g values to multiply by u vector element-wise
        each layer can have a different value of g
        """
        vec_g = np.zeros((np.sum(self.dimensions),))
        for i in range(1, self.layers+1):
            end_range = np.sum(self.dimensions[:i+1])
            start_range = np.sum(self.dimensions[:i])
            vec_g[start_range:end_range] = self.tanh_factors[i-1]
        self.g_vec = vec_g[self.channels*self.image_dim**2:]
    
    def activation_function(self, vec):
        """
        activation function: r = tanh(gu) (eqn.17)
        """
        return np.tanh(self.g_vec*vec)
    
    def neural_dynamics(self, img):
        """
        Neural dynamics for block matrix structure
        Initialise u and r vectors as same size as total dimension of network (sum of dimensions of individual layers)

        i.e. in our 2 layer example:
                    |x|             |0|
            r_vec = |0| and u_vec = |0|
                    |0|             |0|

        Run neural dynamics until every layer has reached convergence criteria, or 3000 update steps have been carried out steps
        Adaptive learning schedule used (i.e. euler step = max(0.2/(1+0.005*T), 0.05)), parameters should be tuned to optimize run-time
        """
        conversion_ticker = 0
        x = img.flatten()
        u_vec = np.zeros(np.sum(self.dimensions))
        r_vec = np.zeros(np.sum(self.dimensions))
        r_vec[:self.channels*self.image_dim**2] = x
        delta = [np.inf]*self.layers
        W_tilda = self.deep_matrix_weights.multiply(self.deep_matrix_structure)+self.deep_matrix_identity
        updates = 0
        while updates < 3000:
            #convergence criteria (||du||/||u|| < 1e-4 for each layer in our network)
            if all(ele < 1e-4 for ele in delta):
                conversion_ticker=1
                break
            #adaptive learning schedule for our euler step
            lr = max((self.euler_step/(1+0.005*updates)), 0.05)
            #calculate du/dT (eqn. 17)
            delta_u = (-u_vec + W_tilda.dot(r_vec))[self.channels*self.image_dim**2:]
            u_vec[self.channels*self.image_dim**2:] += lr*delta_u
            r_vec[self.channels*self.image_dim**2:] = self.activation_function(u_vec[self.channels*self.image_dim**2:])
            updates += 1
            #calculate ||du||/||u|| for each layer
            for layer in range(1, self.layers+1):
                start_token_large = np.sum(self.dimensions[:layer])
                end_token_large = np.sum(self.dimensions[:layer+1])
                start_token_small = int(np.sum(self.dimensions[1:][:layer-1]))
                end_token_small = np.sum(self.dimensions[1:][:layer])
                delta_layer = np.linalg.norm(delta_u[start_token_small:end_token_small])/np.linalg.norm(u_vec[start_token_large:end_token_large])
                delta[layer-1] = delta_layer  
        return r_vec, conversion_ticker
    
    def update_weights(self, r_vec):
        """
        Weight update step (eqn.19)
        Take outer product of complete output (for all layers) and carry out update on whole block weight matrix (previously described for 2 layer case)
        Use weights_adjustment_matrix to take into account 1/1+gamma term in L matrices
        Use weights_update_matrix to take into account that learning rate for L matrices is half that of W matrices
        Adaptive learning schedule used:
            current_lr = max(lr/(1+decay*epoch), lr_floor)
            parameters should be tuned for optimal network performance
        """
        self.current_lr = max(self.lr/(1+self.decay*self.epoch), self.lr_floor)
        update_matrix = np.outer(r_vec, r_vec)
        grad_weights = self.weights_update_matrix.multiply(update_matrix - self.weights_adjustment_matrix.multiply(self.deep_matrix_weights))
        self.deep_matrix_weights += self.current_lr*grad_weights
                
    def training(self, epochs, images):
        """
        Present one image at a time and carry out neural dynamics and weight update (eqn.17 and 19)
        Track conversion of neural dynamics, time taken per epoch etc
        Need to include tracking of cost - did this for single layer case but not yet implemented for deep network
        """
        self.n_images = images.shape[0]
        for epoch in range(epochs):
            img_array = shuffle(images, random_state = epoch)
            epoch_start = time.time()
            sum_ticker = 0
            for img in img_array:
                r, conversion_ticker = self.neural_dynamics(img)
                sum_ticker += conversion_ticker
                self.update_weights(r)
            self.epoch+=1
            epoch_end = time.time()
            epoch_time = epoch_end-epoch_start
            self.conversion_tickers.append(sum_ticker/self.n_images)
            print('Epoch: {0}\nTime_Taken: {1}\nConversion: {2}\nCurrent Learning Rate: {3}\n\n'.format(self.epoch, epoch_time, self.conversion_tickers[-1], self.current_lr))
            