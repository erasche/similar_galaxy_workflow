"""
Predict nodes in graphichal data (Galaxy workflows) using Machine Learning
"""
import sys
import numpy as np
import time
import os

# machine learning library
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import prepare_data


class PredictNextTool:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.current_working_dir = os.getcwd()
        self.network_config_json_path = self.current_working_dir + "/data/model.json"
        self.loss_path = self.current_working_dir + "/data/loss_history.txt"
        self.val_loss_path = self.current_working_dir + "/data/val_loss_history.txt"
        self.epoch_weights_path = self.current_working_dir + "/data/weights/weights-epoch-{epoch:02d}.hdf5"
        self.train_abs_top_pred_path = self.current_working_dir + "/data/train_abs_top_pred.txt"
        self.train_top_compatibility_pred_path = self.current_working_dir + "/data/train_top_compatible_pred.txt"
        self.test_abs_top_pred_path = self.current_working_dir + "/data/test_abs_top_pred.txt"
        self.test_top_compatibility_pred_path = self.current_working_dir + "/data/test_top_compatible_pred.txt"
        self.test_actual_abs_top_pred_path = self.current_working_dir + "/data/test_actual_abs_top_pred.txt"
        self.test_actual_top_compatibility_pred_path = self.current_working_dir + "/data/test_actual_top_compatible_pred.txt"

    @classmethod
    def evaluate_dt( self ):
        """
        Predict using support vector classifier
        """
        print ( "Dividing data..." )
        # get training and test data and their labels
        data = prepare_data.PrepareData()
        train_data, train_labels, test_data, test_labels, test_actual_data, test_actual_labels, dictionary, reverse_dictionary, next_compatible_tools = data.get_data_labels_mat()
        model = DecisionTreeClassifier()
        print( "Training decision tree classifier..." )
        model.fit( train_data, train_labels )
        print ( "Training finished" )
        print( "Predicting..." )
        size = test_labels.shape[ 0 ]
        dimensions = test_labels.shape[ 1 ]
        topk_abs_pred = np.zeros( [ size ] )
        for i in range( size ):
            topk_acc = 0.0
            actual_classes_pos = np.where( test_labels[ i ] > 0 )[ 0 ]
            topk = 1 #len( actual_classes_pos )
            test_sample = np.reshape( test_data[ i ], ( 1, test_data.shape[ 1 ] ) )
            test_sample_pos = np.where( test_data[ i ] > 0 )[ 0 ]
            test_sample_tool_pos = test_data[ i ][ test_sample_pos[ 0 ]: ]
            sample = np.reshape( test_data[ i ], ( 1, test_data[ i ].shape[ 0 ] ) )
            prediction = model.predict_proba( sample )
            #prediction = np.reshape( prediction, ( dimensions, ) )
            print prediction
            prediction_pos = np.argsort( prediction, axis=-1 )
            topk_prediction_pos = prediction_pos[ -topk: ]
            sequence_tool_names = [ reverse_dictionary[ int( tool_pos ) ] for tool_pos in test_sample_tool_pos ]
            actual_next_tool_names = [ reverse_dictionary[ int( tool_pos ) ] for tool_pos in actual_classes_pos ]
            top_predicted_next_tool_names = [ reverse_dictionary[ int( tool_pos ) ] for tool_pos in topk_prediction_pos if int( tool_pos ) != 0 ]
            seq_last_tool = sequence_tool_names[ -1 ]
            next_possible_tools = next_compatible_tools[ seq_last_tool ].split( "," )
            for pred_pos in topk_prediction_pos:
                if pred_pos in actual_classes_pos or reverse_dictionary[ int( pred_pos ) ] in next_possible_tools:
                    topk_acc += 1.0
            topk_acc = topk_acc / float( topk )
            topk_abs_pred[ i ] = topk_acc
            print( "Topk precision: %.2f" % topk_acc )
        print( "Average topk absolute precision: %.2f" % ( np.mean( topk_abs_pred ) ) )

    @classmethod
    def evaluate_mlp( self, dense_units=1024 ):
        """
        Predict using multi-layer perceptron
        """
        print ( "Dividing data..." )
        # get training and test data and their labels
        data = prepare_data.PrepareData()
        train_data, train_labels, test_data, test_labels, test_actual_data, test_actual_labels, dictionary, reverse_dictionary, next_compatible_tools = data.get_data_labels_mat()
        model = MLPClassifier( hidden_layer_sizes=( dense_units, dense_units ), verbose=True, learning_rate='adaptive', batch_size=20, tol=1e-5 )
        print( "Training Multi-layer perceptron..." )
        model.fit( train_data, train_labels )
        print ( "Training finished" )
        print( "Predicting..." )
        predictions = model.predict_proba( test_data )
        self.verify_predictions( test_data, test_labels, predictions, dictionary, reverse_dictionary, next_compatible_tools )

    @classmethod 
    def verify_predictions( self, test_data, test_labels, predictions, dictionary, reverse_dictionary, next_compatible_tools ):
        """
        Compute topk accuracy for each test sample
        """
        size = test_labels.shape[ 0 ]
        dimensions = test_labels.shape[ 1 ]
        topk_abs_pred = np.zeros( [ size ] )
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
            sequence_tool_names = [ reverse_dictionary[ int( tool_pos ) ] for tool_pos in test_sample_tool_pos ]
            actual_next_tool_names = [ reverse_dictionary[ int( tool_pos ) ] for tool_pos in actual_classes_pos ]
            top_predicted_next_tool_names = [ reverse_dictionary[ int( tool_pos ) ] for tool_pos in topk_prediction_pos if int( tool_pos ) != 0 ]
            seq_last_tool = sequence_tool_names[ -1 ]
            next_possible_tools = next_compatible_tools[ seq_last_tool ].split( "," )
            for pred_pos in topk_prediction_pos:
                if pred_pos in actual_classes_pos or reverse_dictionary[ int( pred_pos ) ] in next_possible_tools:
                    topk_acc += 1.0
            topk_acc = topk_acc / float( topk )
            topk_abs_pred[ i ] = topk_acc
        print( "Average topk absolute precision: %.2f" % ( np.mean( topk_abs_pred ) ) )


if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python predict_next_tool.py" )
        exit( 1 )
    start_time = time.time()
    predict_tool = PredictNextTool()
    predict_tool.evaluate_mlp()
    #predict_tool.evaluate_dt()
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
