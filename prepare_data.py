"""
Prepare the workflows to be used by downstream machine learning algorithms
"""

import os
import collections
import numpy as np
import json
import random
import h5py


class PrepareData:

    @classmethod
    def __init__( self, max_seq_length, test_data_share ):
        """ Init method. """
        self.current_working_dir = os.getcwd()
        self.raw_file = self.current_working_dir + "/data/workflow_connections_paths.txt"
        self.data_dictionary = self.current_working_dir + "/data/data_dictionary.txt"
        self.data_rev_dict = self.current_working_dir + "/data/data_rev_dict.txt"
        self.train_file = self.current_working_dir + "/data/train_file.txt"
        self.train_file_sequence = self.current_working_dir + "/data/train_file_sequence.txt"
        self.test_file = self.current_working_dir + "/data/test_file.txt"
        self.test_file_sequence = self.current_working_dir + "/data/test_file_sequence.txt"
        self.train_data_labels_dict = self.current_working_dir + "/data/train_data_labels_dict.json"
        self.test_data_labels_dict = self.current_working_dir + "/data/test_data_labels_dict.json"
        self.train_data_labels_names_dict = self.current_working_dir + "/data/train_data_labels_names_dict.json"
        self.test_data_labels_names_dict = self.current_working_dir + "/data/test_data_labels_names_dict.json"
        self.compatible_tools_filetypes = self.current_working_dir + "/data/compatible_tools.json"
        self.paths_frequency = self.current_working_dir + "/data/workflow_paths_freq.txt"
        self.train_data = self.current_working_dir + "/data/train_data.h5"
        self.test_data = self.current_working_dir + "/data/test_data.h5"
        self.max_tool_sequence_len = max_seq_length
        self.test_share = test_data_share

    @classmethod
    def process_processed_data( self, fname ):
        """
        Get all the tools and complete set of individual paths for each workflow
        """
        tokens = list()
        raw_paths = list()
        with open( fname ) as f:
            data = f.readlines()
        raw_paths = [ x.replace( "\n", '' ) for x in data ]
        for item in raw_paths:
            split_items = item.split( "," )
            for token in split_items:
                if token not in tokens:
                    tokens.append( token )
        tokens = np.array( tokens )
        tokens = np.reshape( tokens, [ -1, ] )
        return tokens, raw_paths

    @classmethod
    def create_data_dictionary( self, words ):
        """
        Create two dictionaries having tools names and their indexes
        """
        count = collections.Counter( words ).most_common()
        dictionary = dict()
        for word, _ in count:
            dictionary[ word ] = len( dictionary ) + 1
        reverse_dictionary = dict( zip( dictionary.values(), dictionary.keys() ) )
        with open( self.data_dictionary, 'w' ) as data_dict:
            data_dict.write( json.dumps( dictionary ) )
        with open( self.data_rev_dict, 'w' ) as data_rev_dict:
            data_rev_dict.write( json.dumps( reverse_dictionary ) )
        return dictionary, reverse_dictionary

    @classmethod
    def decompose_test_paths( self, paths, dictionary, file_pos, file_names ):
        """
        Decompose the paths to variable length sub-paths keeping the first tool fixed
        """
        sub_paths_pos = list()
        sub_paths_names = list()
        for index, item in enumerate( paths ):
            tools = item.split( "," )
            len_tools = len( tools )
            if len_tools <= self.max_tool_sequence_len:
                for window in range( 1, len_tools ):
                    sequence = tools[ 0: window + 1 ]
                    tools_pos = [ str( dictionary[ str( tool_item ) ] ) for tool_item in sequence ]
                    if len( tools_pos ) > 1:
                        sub_paths_pos.append( ",".join( tools_pos ) )
                        sub_paths_names.append( ",".join( sequence ) )
                print( "Path processed: %d" % index )
        sub_paths_pos = list( set( sub_paths_pos ) )
        sub_paths_names = list( set( sub_paths_names ) )
        with open( file_pos, "w" ) as sub_paths_file_pos:
            for item in sub_paths_pos:
                sub_paths_file_pos.write( "%s\n" % item )
        with open( file_names, "w" ) as sub_paths_file_names:
            for item in sub_paths_names:
                sub_paths_file_names.write( "%s\n" % item )
        return sub_paths_pos

    @classmethod
    def take_actual_paths( self, paths, dictionary, file_pos, file_names ):
        """
        Take paths as such. No decomposition.
        """
        sub_paths_pos = list()
        sub_paths_names = list()
        for index, item in enumerate( paths ):
            sequence = item.split( "," )
            if len( sequence ) <= self.max_tool_sequence_len:
                tools_pos = [ str( dictionary[ str( tool_item ) ] ) for tool_item in sequence ]
                if len( tools_pos ) > 1:
                    tools_pos = ",".join( tools_pos )
                    data_seq = ",".join( sequence )
                    sub_paths_pos.append( tools_pos )
                    sub_paths_names.append( data_seq )
        with open( file_pos, "w" ) as sub_paths_file_pos:
            for item in sub_paths_pos:
                sub_paths_file_pos.write( "%s\n" % item )
        with open( file_names, "w" ) as sub_paths_file_names:
            for item in sub_paths_names:
                sub_paths_file_names.write( "%s\n" % item )
        return sub_paths_pos

    @classmethod
    def prepare_paths_labels_dictionary( self, read_file ):
        """
        Create a dictionary of sequences with their labels for training and test paths
        """
        paths = open( read_file, "r" )
        paths = paths.read().split( "\n" )
        paths_labels = dict()
        for item in paths:
            if item and item not in "":
                tools = item.split( "," )
                label = tools[ -1 ]
                train_tools = tools[ :len( tools ) - 1 ]
                train_tools = ",".join( train_tools )
                if train_tools in paths_labels:
                    paths_labels[ train_tools ] += "," + label
                else:
                    paths_labels[ train_tools ] = label
        return paths_labels

    @classmethod
    def randomize_data( self, train_data, train_labels ):
        """
        Randomize the train data after its inflation
        """
        size_data = train_data.shape
        size_labels = train_labels.shape
        rand_train_data = np.zeros( [ size_data[ 0 ], size_data[ 1 ] ] )
        rand_train_labels = np.zeros( [ size_labels[ 0 ], size_labels[ 1 ] ] )
        indices = np.arange( size_data[ 0 ] )
        random.shuffle( indices )
        for index, random_index in enumerate( indices ):
            rand_train_data[ index ] = train_data[ random_index ]
            rand_train_labels[ index ] = train_labels[ random_index ]
        return rand_train_data, rand_train_labels

    @classmethod
    def reconstruct_original_distribution( self, reverse_dictionary, train_data, train_labels ):
        """
        Reconstruct the original distribution in training data
        """
        paths_frequency = dict()
        repeated_train_sample = list()
        repeated_train_sample_label = list()
        train_data_size = train_data.shape[ 0 ]
        with open( self.paths_frequency, "r" ) as frequency:
            paths_frequency = json.loads( frequency.read() )
        for i in range( train_data_size ):
            label_tool_pos = np.where( train_labels[ i ] > 0 )[ 0 ]
            train_sample = np.reshape( train_data[ i ], ( 1, train_data.shape[ 1 ] ) )
            train_sample_pos = np.where( train_data[ i ] > 0 )[ 0 ]
            train_sample_tool_pos = train_data[ i ][ train_sample_pos[ 0 ]: ]
            sample_tool_names = ",".join( [ reverse_dictionary[ int( tool_pos ) ] for tool_pos in train_sample_tool_pos ] )
            label_tool_names = [ reverse_dictionary[ int( tool_pos ) ] for tool_pos in label_tool_pos ]
            for label in label_tool_names:
                reconstructed_path = sample_tool_names + "," + label
                try:
                    freq = int( paths_frequency[ reconstructed_path ] ) - 1
                    if freq > 1:
                        adjusted_freq = int( paths_frequency[ reconstructed_path ] - 1 )
                        tr_data = np.tile( train_data[ i ], ( adjusted_freq, 1 ) )
                        tr_label = np.tile( train_labels[ i ], ( adjusted_freq, 1 ) )
                        repeated_train_sample.extend( tr_data )
                        repeated_train_sample_label.extend( tr_label )
                except Exception as key_error:
                    continue
            print( "Path reconstructed: %d" % i )
        new_data_len = len( repeated_train_sample )
        tr_data_array = np.zeros( [ new_data_len, train_data.shape[ 1 ] ] )
        tr_label_array = np.zeros( [ new_data_len, train_labels.shape[ 1 ] ] )
        for ctr, item in enumerate( repeated_train_sample ):
            tr_data_array[ ctr ] = item
            tr_label_array[ ctr ] = repeated_train_sample_label[ ctr ]
        train_data = np.vstack( ( train_data, tr_data_array  ) )
        train_labels = np.vstack( ( train_labels, tr_label_array ) )
        return train_data, train_labels

    @classmethod
    def verify_overlap( self, train_data, test_data, reverse_dictionary ):
        """
        Verify the overlapping of samples in train and test data
        """
        train_data_size = train_data.shape[ 0 ]
        test_data_size = test_data.shape[ 0 ]
        train_samples = list()
        test_samples = list()
        for i in range( train_data_size ):
            train_sample = np.reshape( train_data[ i ], ( 1, train_data.shape[ 1 ] ) )
            train_sample_pos = np.where( train_data[ i ] > 0 )[ 0 ]
            train_sample_tool_pos = train_data[ i ][ train_sample_pos[ 0 ]: ]
            sample_tool_names = ",".join( [ str(tool_pos) for tool_pos in train_sample_tool_pos ] )
            train_samples.append( sample_tool_names )
        for i in range( test_data_size ):
            test_sample = np.reshape( test_data[ i ], ( 1, test_data.shape[ 1 ] ) )
            test_sample_pos = np.where( test_data[ i ] > 0 )[ 0 ]
            test_sample_tool_pos = test_data[ i ][ test_sample_pos[ 0 ]: ]
            sample_tool_names = ",".join( [ str(tool_pos) for tool_pos in test_sample_tool_pos ] )
            test_samples.append( sample_tool_names )
        intersection = list( set( train_samples ).intersection( set( test_samples ) ) )
        print( "Overlap in train and test: %d" % len( intersection ) )

    @classmethod
    def pad_paths( self, paths_dictionary, num_classes ):
        """
        Add padding to the tools sequences and create multi-hot encoded labels
        """
        size_data = len( paths_dictionary )
        data_mat = np.zeros( [ size_data, self.max_tool_sequence_len ] )
        label_mat = np.zeros( [ size_data, num_classes + 1 ] )
        train_counter = 0
        for train_seq, train_label in list( paths_dictionary.items() ):
            positions = train_seq.split( "," )
            start_pos = self.max_tool_sequence_len - len( positions )
            for id_pos, pos in enumerate( positions ):
                data_mat[ train_counter ][ start_pos + id_pos ] = int( pos )
            for label_item in train_label.split( "," ):
                label_mat[ train_counter ][ int( label_item ) ] = 1.0
            train_counter += 1
        return data_mat, label_mat

    @classmethod
    def get_filetype_compatibility( self, filetypes_path, dictionary ):
        """
        Get the next tools with compatible file types for each tool
        """
        with open( filetypes_path, "r" ) as compatible_tools_file:
            tools_compatibility = json.loads( compatible_tools_file.read() )
        return tools_compatibility

    @classmethod
    def write_to_file( self, file_path, file_names_path, dictionary, reverse_dictionary ):
        """
        Write to file
        """
        path_seq_names = dict()
        with open( file_path, 'w' ) as multilabel_file:
            multilabel_file.write( json.dumps( dictionary ) )
        for item in dictionary:
            path_names = ",".join( [ reverse_dictionary[ int( pos ) ] for pos in item.split( "," ) ] )
            path_label_names = ",".join( [ reverse_dictionary[ int( pos ) ] for pos in dictionary[ item ].split( "," ) ] )
            path_seq_names[ path_names ] = path_label_names
        with open( file_names_path, "w" ) as multilabel_file:
            multilabel_file.write( json.dumps( path_seq_names ) )
        
    @classmethod
    def remove_duplicate_paths( self, train_dict, test_dict ):
        """
        Remove duplicate paths from test paths
        """
        clean_test_dict = dict()
        for path in test_dict:
            if path not in train_dict:
                clean_test_dict[ path ] = test_dict[ path ]
        return clean_test_dict
        
    @classmethod
    def split_test_data( self, test_dict_complete ):
        """
        Split into test and train data randomly for each run
        """
        internal_test_share = 0.5
        test_dict = dict()
        all_test_paths = test_dict_complete.keys()
        random.shuffle( list( all_test_paths ) )
        split_number = int( internal_test_share * len( all_test_paths ) )
        for index, path in enumerate( list( all_test_paths ) ):
            if index < split_number:
                test_dict[ path ] = test_dict_complete[ path ]
        return test_dict

    @classmethod
    def save_as_h5py( self, data, label, file_path ):
        """
        Save the samples and their labels as h5 files
        """
        hf = h5py.File( file_path, 'w' )
        hf.create_dataset( 'data', data=data, compression="gzip", compression_opts=9 )
        hf.create_dataset( 'data_labels', data=label, compression="gzip", compression_opts=9 )
        hf.close()

    @classmethod
    def get_data_labels_mat( self ):
        """
        Convert the training and test paths into corresponding numpy matrices
        """
        processed_data, raw_paths = self.process_processed_data( self.raw_file )
        dictionary, reverse_dictionary = self.create_data_dictionary( processed_data )
        num_classes = len( dictionary )
        print( "Raw paths: %d" % len( raw_paths ) )
        random.shuffle( raw_paths )
        split_number = int( self.test_share * len( raw_paths ) )
        test_paths = raw_paths[ :split_number ]
        train_paths = raw_paths[ split_number: ]
        test_paths = self.decompose_test_paths( test_paths, dictionary, self.test_file, self.test_file_sequence )
        train_paths = self.take_actual_paths( train_paths, dictionary, self.train_file, self.train_file_sequence )

        print( "Train paths: %d" % len( train_paths ) )
        print( "Test paths: %d" % len( test_paths ) )
        print( "Creating dictionaries..." )
        train_paths_dict = self.prepare_paths_labels_dictionary( self.train_file )

        print( "Train data: %d" % len( train_paths_dict ) )
        test_paths_dict = self.prepare_paths_labels_dictionary( self.test_file )

        print( "Test data before removing duplicates: %d" % len( test_paths_dict ) )
        test_paths_dict = self.remove_duplicate_paths( train_paths_dict, test_paths_dict )

        print( "Test data after removing duplicates: %d" % len( test_paths_dict ) )
        test_paths_dict = self.split_test_data( test_paths_dict )

        print( "Test data after size reduction: %d" % len( test_paths_dict ) )
        self.write_to_file( self.test_data_labels_dict, self.test_data_labels_names_dict, test_paths_dict, reverse_dictionary )
        self.write_to_file( self.train_data_labels_dict, self.train_data_labels_names_dict, train_paths_dict, reverse_dictionary )

        print( "Padding paths with 0s..." )
        train_data, train_labels = self.pad_paths( train_paths_dict, num_classes )
        test_data, test_labels = self.pad_paths( test_paths_dict, num_classes )

        print( "Verifying overlap in train and test data..." )
        self.verify_overlap( train_data, test_data, reverse_dictionary )

        #print( "Restoring the original data distribution in training data..." )
        #train_data, train_labels = self.reconstruct_original_distribution( reverse_dictionary, train_data, train_labels )

        print( "Randomizing the train data..." )
        train_data, train_labels = self.randomize_data( train_data, train_labels )

        self.save_as_h5py( train_data, train_labels, self.train_data )
        self.save_as_h5py( test_data, test_labels, self.test_data )

