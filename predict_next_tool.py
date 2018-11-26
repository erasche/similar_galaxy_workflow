"""
Predict next tools in the Galaxy workflows
using machine learning (recurrent neural network)
"""

import sys
import numpy as np
import time
import os
import json
import h5py

# machine learning library
from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers.core import SpatialDropout1D
from keras.optimizers import RMSprop
from keras.models import model_from_json

import extract_workflow_connections
import prepare_data


# file paths
CURRENT_WORKING_DIR = os.getcwd()
NETWORK_C0NFIG_JSON_PATH = CURRENT_WORKING_DIR + "/data/generated_files/model.json"
EPOCH_WEIGHTS_PATH = CURRENT_WORKING_DIR + "/data/weights/weights-epoch-{epoch:d}.hdf5"
DATA_REV_DICT = CURRENT_WORKING_DIR + "/data/generated_files/data_rev_dict.txt"
DATA_DICTIONARY = CURRENT_WORKING_DIR + "/data/generated_files/data_dictionary.txt"
TRAIN_DATA = CURRENT_WORKING_DIR + "/data/generated_files/train_data.h5"
TEST_DATA = CURRENT_WORKING_DIR + "/data/generated_files/test_data.h5"


class PredictNextTool:

    @classmethod
    def __init__( self, epochs ):
        """ Init method. """
        self.n_epochs = epochs
        self.BEST_MODEL_PATH = CURRENT_WORKING_DIR + "/data/weights/weights-epoch-"+ str(epochs) + ".hdf5"

    @classmethod
    def save_network( self, model ):
        """
        Save the network as json file
        """
        with open( NETWORK_C0NFIG_JSON_PATH, "w" ) as json_file:
            json_file.write( model )

    @classmethod
    def read_file( self, file_path ):
        """
        Read a file
        """
        with open( file_path, "r" ) as json_file:
            file_content = json.loads( json_file.read() )
        return file_content

    @classmethod
    def get_h5_data( self, file_name ):
        """
        Read h5 file to get train and test data
        """
        hf = h5py.File( file_name, 'r' )
        return hf.get( "data" ), hf.get( "data_labels" )

    @classmethod
    def evaluate_recurrent_network( self, run, network_config, dictionary, reverse_dictionary, train_data, train_labels, test_data, test_labels ):
        """
        Define recurrent neural network and train sequential data
        """
        print( "Experiment run: %d/%d" % ( ( run + 1 ), network_config[ "experiment_runs" ] ) )

        # Increase the dimension by 1 to mask the 0th position
        dimensions = len( dictionary ) + 1
        optimizer = RMSprop( lr=network_config[ "learning_rate" ] )

        # define the recurrent network
        model = Sequential()
        model.add( Embedding( dimensions, network_config[ "embedding_vec_size" ], mask_zero=True ) )
        model.add( SpatialDropout1D( network_config[ "dropout" ] ) )
        model.add( GRU( network_config[ "memory_units" ], dropout=network_config[ "dropout" ], recurrent_dropout=network_config[ "dropout" ], return_sequences=True, activation=network_config[ "activation_recurrent" ] ) )
        model.add( Dropout( network_config[ "dropout" ] ) )
        model.add( GRU( network_config[ "memory_units" ], dropout=network_config[ "dropout" ], recurrent_dropout=network_config[ "dropout" ], return_sequences=False, activation=network_config[ "activation_recurrent" ] ) )
        model.add( Dropout( network_config[ "dropout" ] ) )
        model.add( Dense( dimensions, activation=network_config[ "activation_output" ] ) )
        model.compile( loss=network_config[ "loss_type" ], optimizer=optimizer )

        # save the network as json
        self.save_network( model.to_json() )
        model.summary()
        
        # create checkpoint after each epoch - save the weights to h5 file
        checkpoint = ModelCheckpoint( EPOCH_WEIGHTS_PATH, verbose=0, mode='max' )
        callbacks_list = [ checkpoint ]

        # fit the model
        print ( "Start training..." )
        model_fit_callbacks = model.fit( train_data, train_labels, batch_size=network_config[ "batch_size" ], epochs=self.n_epochs, callbacks=callbacks_list, shuffle="batch" )
        print ( "Training finished" )
        
    @classmethod
    def load_saved_model( self, network_config_path, weights_path ):
        """
        Load the saved trained model using the saved network and its weights
        """
        with open( network_config_path, 'r' ) as network_config_file:
            loaded_model = network_config_file.read()

        # load the network
        loaded_model = model_from_json(loaded_model)

        # load the saved weights into the model
        loaded_model.load_weights( weights_path )
        return loaded_model
        
    @classmethod
    def verify_model(self, model, x, y, reverse_data_dictionary, dimensions):
        """
        Verify the model on test data
        """
        size = y.shape[ 0 ]
        topk_abs_pred = np.zeros( [ size ] )
        topk_compatible_pred = np.zeros( [ size ] )
        ave_abs_precision = list()
        # loop over all the test samples and find prediction precision
        for i in range( size ):
            actual_classes_pos = np.where( y[ i ] > 0 )[ 0 ]
            topk = len( actual_classes_pos )
            test_sample = np.reshape( x[ i ], ( 1, x.shape[ 1 ] ) )
            test_sample_pos = np.where( x[ i ] > 0 )[ 0 ]
            test_sample_tool_pos = x[ i ][ test_sample_pos[ 0 ]: ]

            # predict next tools for a test path
            prediction = model.predict( test_sample, verbose=0 )

            # remove the 0th position as there is no tool at this index
            prediction = np.reshape(prediction, (dimensions,))
            prediction = prediction[ 1:dimensions ]
            prediction_pos = np.argsort( prediction, axis=-1 )
            topk_prediction_pos = prediction_pos[ -topk: ]

            # read tool names using reverse dictionary
            sequence_tool_names = [ reverse_data_dictionary[ str( int( tool_pos ) ) ] for tool_pos in test_sample_tool_pos ]
            actual_next_tool_names = [ reverse_data_dictionary[ str( int( tool_pos ) ) ] for tool_pos in actual_classes_pos ]
            top_predicted_next_tool_names = [ reverse_data_dictionary[ str( int( tool_pos ) + 1 ) ] for tool_pos in topk_prediction_pos ]

            # find false positives
            false_positives = [ tool_name for tool_name in top_predicted_next_tool_names if tool_name not in actual_next_tool_names ]
            absolute_precision = 1 - ( len( false_positives ) / float( len( actual_next_tool_names ) ) )
            ave_abs_precision.append(absolute_precision)
        return np.mean(ave_abs_precision)


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print( "Usage: python predict_next_tool.py <workflow_file_path> <training_epochs>" )
        exit( 1 )
    start_time = time.time()
    
    n_epochs = int(sys.argv[2])
    retrain = False
    network_config = {
        "experiment_runs": 1,
        "n_epochs": n_epochs,
        "batch_size": 128,
        "dropout": 0.2,
        "memory_units": 128,
        "embedding_vec_size": 128,
        "learning_rate": 0.001,
        "max_seq_len": 25,
        "test_share": 0.20,
        "validation_split": 0.2,
        "activation_recurrent": 'elu',
        "activation_output": 'sigmoid',
        "loss_type": "binary_crossentropy"
    }

    experiment_runs = network_config[ "experiment_runs" ]
    
    # Extract and process workflows
    connections = extract_workflow_connections.ExtractWorkflowConnections(sys.argv[1], retrain)
    connections.read_tabular_file()

    # Process the paths from workflows
    print ( "Dividing data..." )
    data = prepare_data.PrepareData( network_config[ "max_seq_len" ], network_config[ "test_share" ], retrain )
    data.get_data_labels_mat()
    predict_tool = PredictNextTool( n_epochs )
    # get data dictionary
    data_dict = predict_tool.read_file( DATA_DICTIONARY )
    reverse_data_dictionary = predict_tool.read_file( DATA_REV_DICT )

    # get training and test data with their labels
    train_data, train_labels = predict_tool.get_h5_data( TRAIN_DATA )
    test_data, test_labels = predict_tool.get_h5_data( TEST_DATA )

    # execute experiment runs and collect results for each run
    for run in range( experiment_runs ):
        predict_tool.evaluate_recurrent_network( run, network_config, data_dict, reverse_data_dictionary, train_data, train_labels, test_data, test_labels )
        
        loaded_model = predict_tool.load_saved_model( NETWORK_C0NFIG_JSON_PATH, predict_tool.BEST_MODEL_PATH )
        reverse_data_dictionary = predict_tool.read_file(DATA_REV_DICT)
        
    print("Evaluating performance on test data...")
    print("Test data size: %d" % len(test_labels))
    absolute_prec_current_model = predict_tool.verify_model(loaded_model, test_data, test_labels, reverse_data_dictionary, test_labels.shape[1])
    print("Absolute precision on test data using current model is: %0.6f" % absolute_prec_current_model)
    
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
