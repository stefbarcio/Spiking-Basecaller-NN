"""
Functions and classes to read data.
"""
import os
from ont_fast5_api.fast5_interface import get_fast5_file

class ReadData:

    """Contains all the data for a Nanopore read
    
    Attributes:
        read_id (str): id of the read
        run_id (str): id of the run
        filename (str): filename where the read is stored
        offset (int): offset of the read signal
        sampling_rate (float): sampling rate of the run
        scaling (float): range/digitisation
        range (float): range of the raw data
        digitisation (float): digitisation
        mux (int): ?
        channel (str): pore channel
        start (float): time when the read was started to sequence, in seconds
        duration (float): time it took to sequence the read, in seconds
        template_start (float): time when the read was started to sequence, in seconds
        template_duration (float): time it took to sequence the read, in seconds
        raw (np.array::int): raw signal
        scaled (np.array::float): scaled signal
        signal (np.array::float): normalized scaled signal
        basecalls (str): basecalls of the read, optional
        phredq (str): quality score of the basecalls, optional
        segmentation (np.array::object): info regarding the segments of the read by tombo
        start_rel_to_raw (int): datapoint where the aligned read starts
        alignment_info (dict): info regarding the alignment that tombo made
        
    """
    
    def __init__(self, read, filename):

        self.read_id = read.read_id
        try:
            self.run_id = read.get_run_id().decode()
        except AttributeError:
            self.run_id = read.get_run_id()
        self.filename = os.path.basename(read.filename)

        read_attrs = read.handle[read.raw_dataset_group_name].attrs
        channel_info = read.handle[read.global_key + 'channel_id'].attrs

        ## signal normalization stuff
        self.offset = int(channel_info['offset'])
        self.sampling_rate = channel_info['sampling_rate']
        self.scaling = channel_info['range'] / channel_info['digitisation']
        self.range = channel_info['range']
        self.digitisation = channel_info['digitisation']

        ## channel stuff
        self.mux = read_attrs['start_mux']
        try:
            self.channel = channel_info['channel_number'].decode()
        except AttributeError:
            self.channel = channel_info['channel_number']
        self.start = read_attrs['start_time'] / self.sampling_rate
        self.duration = read_attrs['duration'] / self.sampling_rate

        # no trimming
        self.template_start = self.start
        self.template_duration = self.duration
        
        ## signal part
        self.raw = read.handle[read.raw_dataset_name][:]
        
        ## if there are basecalls
        try:
            self.basecalls = read.get_analysis_dataset('Basecall_1D_000', 'BaseCalled_template/Fastq').split('\n')[1]
            self.phredq = read.get_analysis_dataset('Basecall_1D_000', 'BaseCalled_template/Fastq').split('\n')[-2]
        except:
            self.basecalls = None
            self.phredq = None
        
        ## if there is tombo resquiggle info
        try:
            ## this succeeds if the data is put by tombo
            self.segmentation = read.get_analysis_dataset('RawGenomeCorrected_000', 'BaseCalled_template/Events')
            self.start_rel_to_raw = read.get_analysis_attributes('RawGenomeCorrected_000/BaseCalled_template/Events')['read_start_rel_to_raw']
            self.alignment_info = read.get_analysis_attributes('RawGenomeCorrected_000/BaseCalled_template/Alignment')
        except TypeError:
            try:
                ## if the data we put it ourselves we use this
                self.segmentation = read.handle['RawGenomeCorrected_000']['BaseCalled_template/Events'][:]
                self.start_rel_to_raw = read.handle['RawGenomeCorrected_000']['BaseCalled_template/Events'].attrs['read_start_rel_to_raw']
                self.alignment_info = None
            except KeyError:
                self.segmentation = None
                self.start_rel_to_raw = None
                self.alignment_info = None
            
    
    def is_basecalled(self):
        if self.basecalls is None:
            return False
        else:
            return True
    
    def is_resquiggled(self):
        if self.segmentation is None:
            return False
        else:
            return True

def read_fast5(fast5_file, read_ids = None):
    """Extract the data from a fast5 file from the desired reads

    Args:
    	fast5_file (str): file to be read

    Returns:
    	A dictionary with read ids as keys and ReadData objects as
    		values.
    """
    
    read_reads = dict()
    with get_fast5_file(fast5_file, 'r') as f5_fh:
        if read_ids is None:
            for read in f5_fh.get_reads():
                read_reads[read.read_id] = (ReadData(read, fast5_file))
        else:
            if not isinstance(read_ids, list):
                read_ids = [read_ids]
            for read_id in read_ids:
                read = f5_fh.get_read(read_id)
                read_reads[read.read_id] = (ReadData(read, fast5_file))
        
    return read_reads

