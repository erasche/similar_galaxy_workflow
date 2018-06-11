"""
Predict nodes in graphichal data (Galaxy workflows) using Machine Learning
"""

import sys
import numpy as np
import time
import os
import h5py
import json

# machine learning library
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import extract_workflow_connections
import prepare_data


class PredictNextTool:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.current_working_dir = os.getcwd()
        self.network_config_json_path = self.current_working_dir + "/data/model.json"
        self.mean_test_absolute_precision = self.current_working_dir + "/data/mean_test_absolute_precision.txt"
        self.mean_test_compatibility_precision = self.current_working_dir + "/data/mean_test_compatibility_precision.txt"
        self.mean_test_actual_absolute_precision = self.current_working_dir + "/data/mean_test_actual_absolute_precision.txt"
        self.mean_test_actual_compatibility_precision = self.current_working_dir + "/data/mean_test_actual_compatibility_precision.txt"
        self.mean_train_loss = self.current_working_dir + "/data/mean_train_loss.txt"
        self.mean_test_loss = self.current_working_dir + "/data/mean_test_loss.txt"
        self.data_rev_dict = self.current_working_dir + "/data/data_rev_dict.txt"
        self.data_dictionary = self.current_working_dir + "/data/data_dictionary.txt"
        self.compatible_tools_filetypes = self.current_working_dir + "/data/compatible_tools.json"
        self.train_data = self.current_working_dir + "/data/train_data.h5"
        self.test_data = self.current_working_dir + "/data/test_data.h5"

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
    def evaluate_mlp( self, test_data_share, max_seq_len, dense_units=256 ):
        """
        Predict using multi-layer perceptron
        """
        print ( "Dividing data..." )
        # get training and test data and their labels
        model = MLPClassifier( hidden_layer_sizes=( network_config[ "units" ], network_config[ "units" ] ), verbose=True, learning_rate='adaptive', batch_size=network_config[ "batch_size" ], tol=network_config[ "toi" ] )
        print( "Training Multi-layer perceptron..." )
        model.fit( train_data, train_labels )
        print ( "Training finished" )
        print( "Predicting..." )
        predictions = model.predict_proba( test_data )
        topk_abs_pred, topk_compatible_pred = self.verify_predictions( test_data, test_labels, predictions, dictionary, reverse_dictionary, next_compatible_tools )
        np.savetxt( self.mean_test_compatibility_precision, topk_compatible_pred, delimiter="," )
        np.savetxt( self.mean_test_actual_absolute_precision, topk_abs_pred, delimiter="," )

    @classmethod 
    def verify_predictions( self, test_data, test_labels, predictions, dictionary, reverse_dictionary, next_compatible_tools ):
        """
        Compute topk accuracy for each test sample
        """
        size = test_labels.shape[ 0 ]
        dimensions = test_labels.shape[ 1 ]
        topk_abs_pred = np.zeros( [ size ] )
        topk_compatible_pred = np.zeros( [ size ] )
        # loop over all the test samples and find prediction precision
        for i in range( size ):
            topk_acc = 0.0
            actual_classes_pos = np.where( test_labels[ i ] > 0 )[ 0 ]
            topk = len( actual_classes_pos )
            test_sample = np.reshape( test_data[ i ], ( 1, test_data.shape[ 1 ] ) )
            test_sample_pos = np.where( test_data[ i ] > 0 )[ 0 ]
            test_sample_tool_pos = test_data[ i ][ test_sample_pos[ 0 ]: ]
            prediction = predictions[ i ]
            prediction = np.reshape( prediction, ( dimensions, ) )
            prediction_pos = np.argsort( prediction, axis=-1 )
            topk_prediction_pos = prediction_pos[ -topk: ]
            sequence_tool_names = [ reverse_dictionary[ str( int( tool_pos ) ) ] for tool_pos in test_sample_tool_pos ]
            actual_next_tool_names = [ reverse_dictionary[ str( int( tool_pos ) ) ] for tool_pos in actual_classes_pos ]
            top_predicted_next_tool_names = [ reverse_dictionary[ str( int( tool_pos ) + 1 ) ] for tool_pos in topk_prediction_pos ]
            seq_last_tool = sequence_tool_names[ -1 ]
            next_possible_tools = next_compatible_tools[ seq_last_tool ].split( "," )
            for pred_pos in topk_prediction_pos:
                if pred_pos in actual_classes_pos or reverse_dictionary[ int( pred_pos ) ] in next_possible_tools:
                    topk_acc += 1.0
            topk_acc = topk_acc / float( topk )
            topk_abs_pred[ i ] = topk_acc

            topk_compatible_acc = topk_acc
            for pred_pos in topk_prediction_pos:
                if reverse_dictionary[ int( pred_pos ) ] in next_possible_tools:
                    topk_compatible_acc += 1.0 / float( topk )
            topk_compatible_pred[ i ] = topk_compatible_acc
        print( "Average topk absolute precision: %.2f" % ( np.mean( topk_abs_pred ) ) )
        print( "Average topk compatible precision: %.2f" % ( np.mean( topk_compatible_pred ) ) )
        return topk_abs_pred, topk_compatible_pred
        

if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python predict_next_tool.py" )
        exit( 1 )
    network_config = {
        "experiment_runs": 1,
        "n_epochs": 100,
        "units": 128,
        "batch_size": 128,
        "toi": 1.0,
        "learning_rate": 0.001,
        "max_seq_len": 25,
        "test_share": 0.2
    }
    start_time = time.time()
    connections = extract_workflow_connections.ExtractWorkflowConnections()
    connections.read_tabular_file()   
    for run in range( network_config[ "experiment_runs" ] ):
        print ( "Dividing data..." )
        data = prepare_data.PrepareData( network_config[ "max_seq_len" ], network_config[ "test_share" ] )
        data.get_data_labels_mat()
        predict_tool = PredictNextTool()
        train_data, train_labels = predict_tool.get_h5_data( predict_tool.train_data )
        test_data, test_labels = predict_tool.get_h5_data( predict_tool.test_data )
        data_dict = predict_tool.read_file( predict_tool.data_dictionary )
        reverse_data_dictionary = predict_tool.read_file( predict_tool.data_rev_dict )
        next_compatible_tools = predict_tool.read_file( predict_tool.compatible_tools_filetypes )
        predict_tool.evaluate_mlp( network_config, train_data, train_labels, test_data, test_labels, data_dict, reverse_data_dictionary, next_compatible_tools )
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
