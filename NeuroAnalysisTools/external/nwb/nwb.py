"""
Copyright (c) 2015 Allen Institute, California Institute of Technology, 
New York University School of Medicine, the Howard Hughes Medical 
Institute, University of California, Berkeley, GE, the Kavli Foundation 
and the International Neuroinformatics Coordinating Facility. 
All rights reserved.
    
Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following 
conditions are met:
    
1.  Redistributions of source code must retain the above copyright 
    notice, this list of conditions and the following disclaimer.
    
2.  Redistributions in binary form must reproduce the above copyright 
    notice, this list of conditions and the following disclaimer in 
    the documentation and/or other materials provided with the distribution.
    
3.  Neither the name of the copyright holder nor the names of its 
    contributors may be used to endorse or promote products derived 
    from this software without specific prior written permission.
    
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
"""
import sys
import os.path
import shutil
import time
import json
import traceback
import h5py
import copy
import numpy as np
from . import nwbts
from . import nwbep
from . import nwbmo

VERS_MAJOR = 1
VERS_MINOR = 0
VERS_PATCH = 5

__version__ = "%d.%d.%d" % (VERS_MAJOR, VERS_MINOR, VERS_PATCH)
FILE_VERSION_STR = "NWB-%s" % __version__

def get_major_vers():
    return VERS_MAJOR

def get_minor_vers():
    return VERS_MINOR

def get_patch_vers():
    return VERS_PATCH

def get_file_vers_string():
    return FILE_VERSION_STR

def create_identifier(base_string):
    """ Creates an identifying string for the file, hopefully unique 
        in and between labs, based on the supplied base string, the NWB file
        version, and the time the file was created. The base string
        should contain the name of the lab, experimenter and project, or
        some other string that is unique to a given lab
    """
    return base_string + "; " + FILE_VERSION_STR + "; " + time.ctime()

# merge dict y into dict x
def recursive_dictionary_merge(x, y):
    for key in y:
        if key in x:
            if isinstance(x[key], dict) and isinstance(y[key], dict):
                recursive_dictionary_merge(x[key], y[key])
            elif x[key] != y[key]:
                x[key] = y[key]
        else:
            x[key] = y[key]
    return x

def load_json(fname):
    # correct the path, in case calling from remote directory
    fname = os.path.join( os.path.dirname(__file__), fname)
    try:
        with open(fname, 'r') as f:
            jin = json.load(f)
            f.close()
    except IOError:
        print("Unable to load json file '%s'" % fname)
        sys.exit(1)
    return jin

def load_yaml(fname):
    import yaml
    # correct the path, in case calling from remote directory
    fname = os.path.join( os.path.dirname(__file__), fname)
    try:
        with open(fname, 'r') as f:
            jin = yaml.load(f)
            f.close()
    except IOError:
        print("Unable to load json file '%s'" % fname)
        sys.exit(1)
    return jin

def load_spec_file(fname):
    if fname.endswith(".yml"):
        return load_yaml(fname)
    elif fname.endswith(".yaml"):
        return load_yaml(fname)
    else: # try json as default
        return load_json(fname)

def load_spec(custom_spec):
    spec = load_spec_file("spec_file.json")
    ts = load_spec_file("spec_ts.json")
    recursive_dictionary_merge(spec, ts)
    mod = load_spec_file("spec_mod.json")
    recursive_dictionary_merge(spec, mod)
    iface = load_spec_file("spec_iface.json")
    recursive_dictionary_merge(spec, iface)
    gen = load_spec_file("spec_general.json")
    recursive_dictionary_merge(spec, gen)
    ep = load_spec_file("spec_epoch.json")
    recursive_dictionary_merge(spec, ep)
    if len(custom_spec) > 0:
        custom = load_spec_file(custom_spec)
        recursive_dictionary_merge(spec, custom)
    #write_json("fullspec.json", spec)
    return spec

def write_json(fname, js):
    with open(fname, "w") as f:
        json.dump(js, f, indent=2)
        f.close()

# it is too easy to create an object and forget to finalize it
# keep track of when each object is created and finalized, and
#   provide a way to detect when finalization doesnt occur
serial_number = 0
object_register = {}
def register_creation(name):
    global serial_number, object_register
    num = serial_number
    object_register[str(num)] = name
    serial_number += 1
    return num

def register_finalization(name, num):
    global object_register
    if str(num) not in object_register:
        print("Serial number error (SN=%d)" % num)
        print("Object '" + name + "' declared final but was never registered")
        print("Stack trace follows")
        print("-------------------")
        traceback.print_stack()
        sys.exit(1)
    if object_register[str(num)] is None:
        print("Object '" + name + "' finalized multiple times")
        print("Stack trace follows")
        print("-------------------")
        traceback.print_stack()
        sys.exit(1)
    object_register[str(num)] = None

def check_finalization():
    global object_register, serial_number
    err = False
    for k, v in object_register.items():
        if v is not None:
            if not err:
                print("----------------------------------")
                print("Finalization error")
            err = True
            print("    object '"+v+"' was not finalized")
    if err:
        sys.exit(1)
    

# TODO some functionality will be broken on append operations. in particular
#   when an attribute stores a list of links, that list will not be
#   properly updated if new links are created during append  FIXME

class NWB(object):
    """ Represents an NWB file. Calling the NWB constructor creates the file.
        The following arguments are recognized:

            **filename** (text -- mandatory) The name of the to-be-created 
            file

            **identifier** (text -- mandatory) A unique identifier for
            the file, to differentiate it from all other files (even in
            other labs). A suggested way to create the identifier is to
            use a lab-specific string and send it to
            nwb.create_identifier(string). This function returns the
            supplied string appended by the present date

            **description** (text -- mandatory) A one or two sentence
            description of the experiment and what the data in the file
            represents

            *start_time* (text -- optional) This is the starting time of the 
            experiment.  If this isn't provided, the start time of the 
            experiment will default to the time that the file is created

            *modify* (boolean -- optional) Opens the file in append mode
            if the file exists. If the file exists and this flag (or
            'overwrite') is not set, an error occurs (to prevent
            accidentally overwriting or modifying an existing file)

            *overwrite* (boolean -- optional) If the specified file exists,
            it will be overwritten

            *keep_original* (boolean -- optional) -- If true, a back-up copy 
            of the original file will be kept, named '<filename>.prev'

            *auto_compress* (boolean -- optional) Data is compressed
            automatically through the API. Setting 'auto_compress=False'
            disables this behavior

            *custom_spec* (text -- optional) A json, yaml or toml file
            used to customize the format specification (pyyaml or toml
            must be installed to use those formats)
    """
    def __init__(self, **vargs):
        self.read_arguments(**vargs)
        self.file_pointer = None
        # make a list to keep track of all time series
        self.ts_list = []
        # list of defined epochs
        self.epoch_list = []
        # module list
        self.modules = []
        # record of all tags used in epochs
        # use a dict as it's easier to filter out dups
        self.epoch_tag_dict = {}
        # load specification
        self.spec = load_spec(self.custom_spec)
        # flag to keep backup of original file, using ".prev" suffix
        self.keep_original = False 
        #
        self.tmp_name = self.file_name + ".tmp"
        if self.file_exists:
            # file exists -- see if modify flag set
            if "modify" in vargs and vargs["modify"] == True:
                self.open_existing()
            elif "overwrite" in vargs and vargs["overwrite"] == True:
                self.create_file()
            elif "keep_original" in vargs and vargs["keep_original"]:
                self.keep_original = True
            else:
                print("File '%s' already exists. Specify 'modify=True' to open for writing" % self.file_name)
                sys.exit(1)
        else:
            # create new file
            self.create_file()
        # when TS datasets are HDF5 linked, keep track of those links and
        #   add them to each TS so file reader knows what data is shared
        # these are data structures to track associations. the lists
        #   store src-dest pairs, indexed by a label, and the maps
        #   store label values indexed by src & dest names
        # keep separate lists for data[] and timestamps[]
        self.ts_data_link_map = {}
        self.ts_data_link_cnt = 0
        self.ts_data_link_lists = {}
        self.ts_time_link_map = {}
        self.ts_time_link_cnt = 0
        self.ts_time_link_lists = {}
        # to track softlinks
        self.ts_time_softlinks = {}
        ## undocumented feature -- automatically close file on exit
        ## this is to avoid case where user forgets to call 'close()'
        ##   and can't figure out why resulting file is broken
        #import atexit
        #atexit.register(self.close)
        self.is_open = True
        self.error_flag = False
        # users may specify different spellings of dtypes than the library
        #   expects. here's a table to handle conversions from some common
        #   forms
        self.dtype_glossary = {}
        self.dtype_glossary["float32"] = 'f4'
        self.dtype_glossary["float4"] = 'f4'
        self.dtype_glossary["float64"] = 'f8'
        self.dtype_glossary["float8"] = 'f8'
        self.dtype_glossary["text"] = 'str'
        self.dtype_glossary["i8"] = 'int64'
        self.dtype_glossary["i4"] = 'int32'
        self.dtype_glossary["i2"] = 'int16'
        self.dtype_glossary["i1"] = 'int8'
        self.dtype_glossary["u8"] = 'uint64'
        self.dtype_glossary["u4"] = 'uint32'
        self.dtype_glossary["u2"] = 'uint16'
        self.dtype_glossary["u1"] = 'uint8'
        self.dtype_glossary["byte"] = 'uint8'

    # internal API function to process constructor arguments
    def read_arguments(self, **vargs):
        err_str = ""
        # file name
        if "filename" in vargs:
            self.file_name = vargs["filename"]
        elif "file_name" in vargs:
            self.file_name = vargs["file_name"]
        else:
            err_str += "    argument '%s' was not specified\n" % "filename"
        # see if file exists -- some arguments aren't required if so
        if os.path.isfile(self.file_name):
            self.file_exists = True
        else:
            self.file_exists = False
        # read start time
        if "start_time" in vargs:
            self.start_time = vargs["start_time"]
            del vargs["start_time"]
        elif "starting_time" in vargs:
            self.start_time = vargs["starting_time"]
            del vargs["starting_time"]
        else:
            self.start_time = time.ctime()
        if "auto_compress" in vargs:
            self.auto_compress = vargs["auto_compress"]
        else:
            self.auto_compress = True
        # allow user to specify custom json specification file
        # when the request to specify multiple files comes in, allow
        #   multiple files to be submitted as a dictionary or list
        if "custom_spec" in vargs:
            self.custom_spec = vargs["custom_spec"]
        else:
            self.custom_spec = []
        # read identifier
        if "identifier" in vargs:
            self.file_identifier = vargs["identifier"]
        elif not self.file_exists:
            err_str += "    argument '%s' was not specified\n" % "identifier"
        # read session description
        if "description" in vargs:
            self.session_description = vargs["description"]
        elif not self.file_exists:
            err_str += "    argument 'description' was not specified\n"
        # handle errors
        if len(err_str) > 0:
            print("Error creating Borg object - missing constructor value(s)")
            print(err_str)
            sys.exit(1)

    def fatal_error(self, msg):
        self.error_flag = True
        print("Error: " + msg)
        print("Stack trace follows")
        print("-------------------")
        traceback.print_stack()
        sys.exit(1)

    # internfal function that stores all tags that were used in epochs,
    #   so that a global list can be appended to the epoch folder on 
    #   file closing
    def add_epoch_tags(self, tags):
        for i in range(len(tags)):
            tag = tags[i]
            if tag not in self.epoch_tag_dict:
                self.epoch_tag_dict[tag] = tag

    ####################################################################
    ####################################################################
    # File operations

    # internal function that creates a new file, including file skeleton
    def create_file(self):
        # open file
        try:
            self.file_pointer = h5py.File(self.tmp_name, "w")
        except IOError:
            print("Unable to create output file '%s'" % self.tmp_name)
            sys.exit(1)
        ################################################################
        # create skeleton
        # make local copy of file pointer
        fp = self.file_pointer
        # create top-level datasets
        fp.create_dataset("nwb_version", data=FILE_VERSION_STR)
        fp.create_dataset("identifier", data=self.file_identifier)
        fp.create_dataset("session_description", data=self.session_description)
        cur_time = time.ctime()
        dt = h5py.special_dtype(vlen=bytes)
        fp.create_dataset("file_create_date", data=[np.string_(cur_time)], maxshape=(None,), chunks=True, dtype=dt)
        fp.create_dataset("session_start_time", data=np.string_(self.start_time))
        # create file skeleton
        hgen = fp.create_group("general")
        self.hgen = hgen
        self.hgen_dev = hgen.create_group("devices")
        #
        hacq = fp.create_group("acquisition")
        hacq.create_group("timeseries")
        hacq.create_group("images")
        #
        hstim = fp.create_group("stimulus")
        hstim.create_group("templates")
        hstim.create_group("presentation")
        #
        fp.create_group("epochs")
        #
        fp.create_group("processing")
        #
        fp.create_group("analysis")


    # internal function to open existing file for writing
    # TODO read timeseries data/time links. add to kernel's link tracking
    def open_existing(self):
        # make backup copy before modifying anything
        # copy2 preserves file metadata (eg, create date)
        shutil.copy2(self.file_name, self.tmp_name)
        # open tmp file for appending
        try:
            self.file_pointer = h5py.File(self.tmp_name, "a")
        except IOError:
            print("Unable to open temp output file '%s'" % self.tmp_name)
            sys.exit(1)
        fp = self.file_pointer
        # TODO verify version
        # append timestamp to file create date's modification attribute
        # modification_time is deprecated. this will be removed in a
        #   future version. keep history of modification times in 
        #   file_create_date
        new_time = time.ctime()
        create = fp["file_create_date"]
        entries = len(create)
        create.resize((entries+1,))
        create[entries] = new_time

        mod_time = []
        if "modification_time" in create.attrs:
            mod_time = create.attrs["modification_time"]
        if isinstance(mod_time, (np.ndarray)):
        #if type(mod_time).__name__ == "ndarray":
            mod_time = mod_time.tolist()
        mod_time.append(np.string_(new_time))
        create.attrs["modification_time"] = mod_time

    # internal function that pushes metadata to the file on file closing
    def write_metadata(self):
        grp = self.file_pointer["general"]
        spec = self.spec["General"]
        self.write_datasets(grp, "", spec)

    def close(self):
        """ Finishes and closes an NWB file. This includes writing pending
            data to disk and adding annotations.
            NOTE: this procedure must be called to produce a valid NWB file

            Arguments:
                None

            Returns:
                Nothing
        """
        # if fatal error flag set, don't both with cleaning things up
        # quit gracefully quietly, otherwise errors here may mask 
        #   previous real problems
        if self.error_flag:
            self.file_pointer.close()
            return
        if not self.is_open:
            return
        self.is_open = False
        # finalize all time series
        # this will be a no-op for series that have already been finalized
        for i in range(len(self.ts_list)):
            self.ts_list[i].finalize()
        # after time series are finalized, go back and document links
        # TODO check to see if data_link and timestamp_link exist
        for k, lst in self.ts_data_link_lists.items():
            for i in range(len(lst)):
                self.file_pointer[lst[i]].attrs["data_link"] = lst
                #self.file_pointer[lst[i]].attrs["data_link"] = np.string_(lst) # VALIDATOR
        for k, lst in self.ts_time_link_lists.items():
            for i in range(len(lst)):
                obj = self.file_pointer[lst[i]]
                obj.attrs["timestamp_link"] = lst
                #obj.attrs["timestamp_link"] = np.string_(lst)
        #for k, lnk in self.ts_time_softlinks.items():
        #    self.file_pointer[k].attrs["data_softlink"] = np.string_(lnk)
        # TODO finalize all modules
        # finalize epochs and write epoch tag list to epoch group
        for i in range(len(self.epoch_list)):
            self.epoch_list[i].finalize()
        tags = []
        for k in self.epoch_tag_dict:
            tags.append(k)
        self.file_pointer["epochs"].attrs["tags"] = np.string_(tags)
        # make sure there are no registered objects that aren't finalized
        check_finalization()
        # write out metadata
        self.write_metadata()
        # close file
        self.file_pointer.close()
        # replace orig w/ tmp
        # keep old file around with suffix '.prev'
        if self.keep_original and os.path.isfile(self.file_name):
            shutil.move(self.file_name, self.file_name + ".prev")
        shutil.move(self.tmp_name, self.file_name)

    ####################################################################
    ####################################################################
    # Link management

    # internal API function to store a link between timeseries::data
    #   so that a summary of all links can be produced when the file
    #   closes
    def record_timeseries_data_link(self, src, dest):
        # make copies of values using shorter names to keep line length down
        label_map = self.ts_data_link_map
        label_lists = self.ts_data_link_lists
        cnt = self.ts_data_link_cnt
        # call storage method for ts::data[]
        n = self.store_timeseries_link(src, dest, label_map, label_lists, cnt)
        # update label counter
        self.ts_data_link_cnt = n

    # internal API function to store a link between timeseries::timestamps
    #   so that a summary of all links can be produced when the file
    #   closes
    def record_timeseries_time_link(self, src, dest):
        # make copies of values using shorter names to keep line length down
        label_map = self.ts_time_link_map
        label_lists = self.ts_time_link_lists
        cnt = self.ts_time_link_cnt
        # call storage method for ts::timestamps[]
        n = self.store_timeseries_link(src, dest, label_map, label_lists, cnt)
        # update label counter
        self.ts_time_link_cnt = n

    # internal API function to store a link between timeseries::timestamps
    #   so that a summary of all links can be produced when the file
    #   closes
    def record_timeseries_data_soft_link(self, src, dest):
        self.ts_time_softlinks[src] = dest

    # internal API function to manage timeseries link indexing
    #
    # this function takes the source and destination paths to objects
    #   that are linked and updates a map of what is linked to what
    # it handles the case where a link is itself linked to a link, and
    #   where the links aren't necessarily defined in order
    # the problem is essentially one of taking edges of N independent 
    #   graphs, one-by-one and in any order, and reconstructing the list
    #   of which edges are in which graph, and labeling the graphs uniquely
    def store_timeseries_link(self, src, dest, label_map, label_lists, cnt):
        # if neither graph vertex is known to be part of a graph, define
        #   a new graph
        if src not in label_map and dest not in label_map:
            label = "graph_%d" % cnt
            # record that this graph has these two vertices
            label_lists[label] = [src, dest]
            cnt = cnt + 1
            # store which graph each vertex is a part of
            label_map[src] = label
            label_map[dest] = label
        elif src in label_map and dest in label_map:
            # both vertices are part of known graphs
            # see if they are both part of the same graph
            if label_map[src] != label_map[dest]:
                # they belong to different graphs
                # these graphs are now connected and need to be merged
                # select one graph to merge the other into it
                # merge smaller into larger
                len_src = len(label_lists[label_map[src]])
                len_dest = len(label_lists[label_map[dest]])
                if len_src > len_dest:
                    label = label_map[src]
                    old_label = label_map[dest] # name of retired graph
                else:
                    label = label_map[dest]
                    old_label = label_map[src]  # name of retired graph
                # retire 2nd graph. reset values using its name to new 
                #   use the name of the other (larger) graph
                # append entries from 2nd graph to list for 1st
                old_list = label_lists[old_label]
                for i in range(len(old_list)):
                    name = old_list[i]
                    label_map[name] = label
                    label_lists[label].append(name)
                # remove record of retired graph
                del label_lists[old_label]
        elif src in label_map:
            # one vertex part of known graph -- add 2nd vertex to that graph
            label = label_map[src]
            label_map[dest] = label
            label_lists[label].append(dest)
        elif dest in label_map:
            # one vertex part of known graph -- add 2nd vertex to that graph
            label = label_map[dest]
            label_map[src] = label
            label_lists[label].append(src)
        else:
            assert "Internal error"
        # return graph count (this will have changed if a new graph was
        #   defined)
        return cnt

    ####################################################################
    ####################################################################
    # create file content

    def create_epoch(self, name, start, stop):
        """ Creates a new Epoch object. Epochs are used to track intervals
            in an experiment, such as exposure to a certain type of stimuli
            (an interval where orientation gratings are shown, or of 
            sparse noise) or a different paradigm (a rat exploring an 
            enclosure versus sleeping between explorations)

            Arguments:
                *name* (text) The name of the epoch, as it will appear in
                the file

                *start* (float) The starting time of the epoch

                *stop* (float) The ending time of the epoch

            Returns:
                Epoch object
        """
        spec = self.spec["Epoch"]
        epo = nwbep.Epoch(name, self, start, stop, spec)
        self.epoch_list.append(epo)
        epo.serial_num = register_creation("Epoch -- " + name)
        return epo

    def create_timeseries(self, ts_type, name, modality="other"):
        """ Creates a new TimeSeries object. Timeseries are used to
            store and associate data or events with the time the
            data/events occur.

            Arguments:
                *ts_type* (text) The type of timeseries to be created.
                Default options are:

                    TimeSeries -- simple time series

                    AbstractFeatureSeries -- features of a presented
                    stimulus. This is particularly useful when storing
                    the raw stimulus is impractical and only certain
                    features of the stimulus are salient. An example is
                    the visual stimulus of orientation gratings, where
                    the phase, spatial/temporal frequency and contrast
                    are relevant, but the individual video frames are
                    impractical to store, and not as useful

                    AnnotationSeries -- stores strings that annotate
                    events or actions, plus the time the annotation was made

                    ElectricalSeries -- Voltage acquired during extracellular
                    recordings

                    ImageSeries -- storage object for 2D image data. An
                    ImageSeries can represent image data within the file
                    or can point to an image stack in an external file
                    (eg, png or tiff)

                    IndexSeries -- series that is composed of samples
                    in an existing time series, for example images that
                    are pulled from an image stack in random order

                    ImageMaskSeries -- a mask that is applied to a visual
                    stimulus

                    IntervalSeries -- a list of starting and stop times
                    of events

                    OpticalSeries -- a series of image frames, such as for
                    video stimulus or optical recording
                    
                    OptogeneticSeries -- optical stimulus applied during
                    an optogentic experiment

                    RoiResponseSeries -- responses of a region-of-interest
                    during optical recordings, such as florescence or dF/F

                    SpatialSeries -- storage of points in space over time

                    SpikeEventSeries -- snapshots of spikes events in
                    an extracellular recording

                    TwoPhotonSeries -- Image stack recorded from a 
                    2-photon microscope

                    VoltageClampSeries, CurrentClampSeries -- current or
                    voltage recurded during a patch clamp experiment

                    VoltageClampStimulusSeries, CurrentClampStimulusSeries
                    -- voltage or current used as stimulus during a
                    patch clamp experiment

                    WidefieldSeries -- Image stack recorded from wide-field
                    imaging

                *name* (text) the name of the TimeSeries, as it will
                appear in the file

                *modality* (text) this indicates where in the file the
                TimeSeries will be stored. Values are:

                    'acquisition' -- acquired data stored under 
                    /acquisition/timeseries

                    'stimulus' -- stimulus data stored under
                    /stimulus/presentations

                    'template' -- a template for a stimulus, useful if
                    a stimulus will be repeated as it only has to be
                    stored once

                    'other' (DEFAULT) -- TimeSeries is to be used in a 
                    module, in which case the module will manage its
                    placement, or it's up to the user where to place it

            Returns:
                TimeSeries object
        """
        # find time series by name
        # recursively examine spec and create dict of required fields
        ts_defn, ancestry = self.create_timeseries_definition(ts_type, [], None)
        if "_value" not in ts_defn["_attributes"]["ancestry"]:
            ts_defn["_attributes"]["ancestry"]["_value"] = []
        ts_defn["_attributes"]["ancestry"]["_value"] = ancestry
        if ts_type == "AnnotationSeries":
            ts = nwbts.AnnotationSeries(name, modality, ts_defn, self)
        elif ts_type == "AbstractFeatureSeries":
            ts = nwbts.AbstractFeatureSeries(name, modality, ts_defn, self)
        else:
            ts = nwbts.TimeSeries(name, modality, ts_defn, self)
        self.ts_list.append(ts)
        ts.serial_num = register_creation(ts_type + " -- " + name)
        return ts

    # internal API call to get specification of time series from config file
    # read spec to create time series definition. do it recursively 
    #   if time series are subclassed
    def create_timeseries_definition(self, ts_type, ancestry, defn):
        ts_dict = self.spec["TimeSeries"]
        if ts_type not in ts_dict:
            self.fatal_error("'%s' is not a recognized time series" % ts_type)
        if defn is None:
            defn = ts_dict[ts_type]
        defn = copy.deepcopy(ts_dict[ts_type])
        # pull in data from superclass
        if "_superclass" in defn:
            # avoid infinite loops in specification
            if ts_type == defn["_superclass"]:
                self.fatal_error("Infinite loop in spec for TS " + ts_type)
            parent = defn["_superclass"]
            del defn["_superclass"]
            # add parent definition to this
            par, ancestry = self.create_timeseries_definition(parent, ancestry, defn)
            defn = recursive_dictionary_merge(par, defn)
        # make ancestry record
        # string cast is necessary because sometimes string is unicode (why??)
        ancestry.append(str(ts_type))
        return defn, ancestry

    def create_module(self, name):
        """ Creates a Module object of the specified name. Interfaces can
            be created by the module and will be stored inside it

            Arguments:
                *name* (text) Name of the module as it will appear in the
                file (under /processing/)

            Returns:
                Module object
        """
        mod = nwbmo.Module(name, self, self.spec["Module"])
        self.modules.append(mod)
        mod.serial_num = register_creation("Module -- " + name)
        return mod

    def set_metadata(self, key, value, **attrs):
        """ Creates a field under /general/ and stores the specified
            information there. 
            NOTE: using the constants defined in nwbco.py is strongly
            encouraged, as this will help prevent accidental typos
            and will not require the user to remember where a particular
            piece of data is to be stored

            Arguments:
                *key* (text) Name of the metadata field. Please use the
                constants and functions defined in nwbco.py

                *value* Value of the data to be stored. This will be text
                in most cases

                *attrs* (dictionary, or key/value pairs) Attributes that
                will be created on the metadata field

            Returns:
                nothing
        """
        if type(key).__name__ == "function":
            self.fatal_error("Function passed instead of string or constant -- please see documentation for usage of '%s'" % key.__name__)
        # metadata fields are specified using hdf5 path
        # all paths relative to general/
        toks = key.split('/')
        # get specification and store data in appropriate slot
        spec = self.spec["General"]
        n = len(toks)
        for i in range(n):
            if toks[i] not in spec:
                # support optional fields/directories
                # recurse tree to find appropriate element
                if i == n-1 and "[]" in spec:
                    spec[toks[i]] = copy.deepcopy(spec["[]"])   # custom field
                    spec = spec[toks[i]]
                elif i < n-1 and "<>" in spec:
                    # variably named group
                    spec[toks[i]] = copy.deepcopy(spec["<>"])
                    spec = spec[toks[i]]
                else:
                    self.fatal_error("Unable to locate '%s' of %s in specification" % (toks[i], key))
            else:
                spec = spec[toks[i]]
        self.check_type(key, value, spec["_datatype"])
        spec["_value"] = value
        # handle attributes
        if "_attributes" not in spec:
            spec["_attributes"] = {}
        for k, v in attrs.items():
            if k not in spec["_attributes"]:
                spec["_attributes"][k] = {}
            fld = spec["_attributes"][k]
            fld["_datatype"] = 'str'
            fld["_value"] = str(v)

    def set_metadata_from_file(self, key, filename, **attrs):
        """ Creates a field under /general/ and stores the contents of
            the specified file in that field
            NOTE: using the constants defined in nwbco.py is strongly
            encouraged, as this will help prevent accidental typos
            and will not require the user to remember where a particular
            piece of data is to be stored

            Arguments:
                *key* (text) Name of the metadata field. Please use the
                constants and functions defined in nwbco.py

                *filename* (text) Name of file containing the data to 
                be stored

                *attrs* (dictionary, or key/value pairs) Attributes that
                will be created on the metadata field

            Returns:
                nothing
        """
        try:
            f = open(filename, 'r')
            contents = f.read()
            f.close()
        except IOError:
            self.fatal_error("Error opening metadata file " + filename)
        self.set_metadata(key, contents, **attrs)

    def create_reference_image(self, stream, name, fmt, desc, dtype=None):
        """ Adds documentation image (or movie) to file. This is stored
            in /acquisition/images/.

            Args:
                *stream* (binary) Data stream of image (eg, binary contents of .png image)

                *name* (text) Name that image will be stored as

                *fmt* (text) Format of the image (eg, "png", "avi")

                *desc* (text) Descriptive text describing the image

                *dtype* (text) Optional field specifying the h5py datatype to use to store *stream*

            Returns:
                *nothing*
        """
        fp = self.file_pointer
        img_grp = fp["acquisition/images"]
        if name in img_grp:
            self.fatal_error("Reference image %s alreayd exists" % name)
        if dtype is None:
            img = img_grp.create_dataset(name, data=stream)
        else:
            img = img_grp.create_dataset(name, data=stream, dtype=dtype)
        img.attrs["format"] = np.string_(fmt)
        img.attrs["description"] = np.string_(desc)
        

    ####################################################################
    # HDF5 interface

    # Internal procedure to verify that value is expected type.
    # Throws assertion if value type is unrecognized or if it's not 
    # convertable to desired type. Procedure fails ungracefully on
    # type error
    def check_type(self, key, value, dtype):
        # TODO verify that value is compatible w/ spec type
        if dtype is None or dtype == "unrestricted":
            return  # implicit OK
        while isinstance(value, (list, np.ndarray)):
            if len(value) == 0:
                msg = "attempted to store empty list (key=%d)" % key
                self.fatal_error(msg)
            value = value[0]
        try:
            if dtype == "str":
                if type(value).__name__ == 'unicode':
                    value = str(value)
                if type(value).__name__ != dtype:
                    m1 = "field '%s' has invalid type\n" % key 
                    m2 = "Expected '%s', found '%s'" % (dtype, type(value).__name__)
                    self.fatal_error(m1 + m2)
            elif dtype.startswith('f'):
                # check for type conversion error
                if isinstance(value, str):
                    raise ValueError
            elif dtype.startswith('uint') or dtype.startswith('int'):
                # check for type conversion error
                if isinstance(value, str):
                    raise ValueError
            elif dtype != "unspecified":
                self.fatal_error("unexpected type: '%s'" % dtype)
        except ValueError:
            m1 = "Type conversion error\n"
            m2 = "Expected '%s', found '%s'" % (dtype, type(value).__name__)
            # fail ungracefully and print stack trace
            self.fatal_error(m1 + m2)

    # internal API function to set key-value pair structure in the 
    #   provided specification
    def set_value_internal(self, key, value, spec, name, dtype=None, **attrs):
        if isinstance(value, nwbts.TimeSeries):
            value = value.full_path()
        if isinstance(value, str):
            value = str(value)
        if dtype is not None and dtype in self.dtype_glossary:
            dtype = self.dtype_glossary[dtype]
        if value is None:
            if dtype is None:
                self.fatal_error("Attempted to set 'None' value in field '%s' (from %s) with no specified dtype" % (key, name))
            if dtype == 'f4' or dtype == 'f8':
                value = float('nan')
            elif dtype == 'str':
                value = ""
            else:
                self.fatal_error("Don't know how to handle None value when dtype is integer")
        # get field definition
        if key not in spec:
            # custom field. make sure it's acceptable
            if "[]" not in spec:
                m1 = "Attempted to add unsupported custom field to '%s'\n"%name
                m2 = "\tkey = " + str(key) + "\n"
                m3 = "\tvalue = " + str(value) + "\n"
                self.fatal_error(m1 + m2 + m3)
            field = copy.deepcopy(spec["[]"])
            spec[key] = field   # add custom field to local spec
        else:
            field = spec[key]   # get known field from local spec
        ########################################
        # reset to dtype if specified by user
        if dtype is not None:
            field["_datatype"] = dtype
        # if expected dtype is NWB object that means we need a link
        # value must be a string (the path) or that type of object
        # handle type checking for this contingency here
        ftype = field["_datatype"]
        if ftype == "interface" or ftype == "timeseries":
            path = ""
            if ftype == "interface" and isinstance(value, nwbmo.Interface):
                path = value.full_path()
            elif ftype == "timeseries" and isinstance(value, nwbts.TimeSeries):
                path = value.full_path()
            elif isinstance(value, str):
                path = str(value)
            else:
                self.fatal_error("Expected type %s, got %s" % (ftype, type(value)))
            if "_value" in field:
                self.fatal_error("Setting link after value already defined")
            field["_value_hardlink"] = path
            # all done here
            return
        if field["_datatype"] == "unrestricted":
            if isinstance(value, str):
                field["_datatype"] = "str"
        # verify type, or set it if it's unrestricted and a float
        if field["_datatype"] == "unrestricted":
            # descend into list, if multi-dimensional
            val = value
            loops = 0
            while isinstance(val, (list, np.ndarray)) and len(val)>0:
                val = val[0]
                loops += 1;
                if loops >= 10:
                    self.fatal_error("Sanity check failed determining type -- please explicitly set dtype when setting this value (%s)", key)
            # set non-dtyped float as float32
            if isinstance(val, (float, np.float64, np.float32)):
                field["_datatype"] = 'f4'
        else:
            self.check_type(key, value, field["_datatype"])
        field["_value"] = value
        for k in list(attrs.keys()):
            if k not in field["_attributes"]:
                self.fatal_error("Custom attributes not supported -- '%s' is not part of field '%s'" %(k, key))
            spec_type = field["_attributes"][k]["_datatype"]
            self.check_type(k, attrs[k], spec_type)
            # use numpy's handling of strings for 'str' as it's more robust
            if spec_type == "str":
                field["_attributes"][k]["_value"] = np.string_(attrs[k])
            else:
                field["_attributes"][k]["_value"] = attrs[k]

    # internal function to add attributes, identified in the supplied spec,
    #   to an existing HDF5 group
    def write_attributes(self, grp, spec):
        attr = spec["_attributes"]
        for k in attr:
            if k.startswith('_'):
                continue    # internal field -- nothing to write out
            if k == "<>" or k == "[]":
                continue    # template
            if "_value" in attr[k]:
                try:
                    # python 3 has different string handling
                    # attribute writing restructured in attempt to
                    #   compensate. If attribute is a list of strings
                    #   then convert each to be a numpy string_
                    #if sys.version_info >= (3, 0):
                    x = attr[k]["_value"]
                    if isinstance(x, (list)) and len(x) > 0:
                        while isinstance(x, (list)):
                            x = x[0]
                        if isinstance(x, (str)):
                            # only handle array of lists
                            if not isinstance(x[0], (str)):
                                self.fatal_error("Multi-dimensional text arrays not presently supported in attributes (attr=%s)" % attr);
                            # convert eacy string
                            y = []
                            x = attr[k]["_value"]
                            for i in range(len(x)):
                                y.append(np.string_(x[i]))
                            attr[k]["_value"] = y
                    elif isinstance(x, (str)):
                        attr[k]["_value"] = np.string_(x)
                    grp.attrs[k] = attr[k]["_value"]
                except TypeError:
                    print("*** Type error ***")
                    print("Attribute " + k)
                    print(attr[k]["_value"])
                    print(type(attr[k]["_value"]))
                    print(len(attr[k]["_value"]))
                    raise
                except ValueError:
                    print("*** Value error ***")
                    print("Attribute " + k)
                    print(attr[k]["_value"])
                    print(type(attr[k]["_value"]))
                    print(len(attr[k]["_value"]))
                    raise

    # internal API function to create a dataset in the specified path,
    #   relative to the specified group. dataset is described in spec
    # different cases for value, hard-link and soft-link are handled
    #   separately
    #
    # grp -- hdf5 group object that datasets are stored under
    # spec -- specification dictionary of data to be stored
    # path -- path under grp that spec applies to (nested groups
    #     call write_datasets recursively -- this is the path to
    #     where things are at a particular recursion round)
    def write_datasets(self, grp, path, spec):
        # write out all fields that have a _value
        for k in spec:
            if k.startswith('_'):
                continue    # internal field -- nothing to write out
            if k == "<>" or k == "[]":
            #if k == "<>" or k == "[]" or k == "{}":
                continue    # template
            # create dataset for fields in spec where _value* is specified
            local_spec = spec[k]
            if local_spec["_datatype"] == "group":
                nest = path + k + "/"
                self.write_datasets(grp, nest, local_spec)
                if "_attributes" in local_spec:
                    self.write_attributes(grp[nest], local_spec)
            elif "_value" in local_spec:
                self.write_dataset_to_file(grp, path, k, local_spec)
            elif "_value_softlink" in local_spec:
                self.write_dataset_as_softlink(grp, path, k, local_spec)
            elif "_value_hardlink" in local_spec:
                self.write_dataset_as_hardlink(grp, path, k, local_spec)
            if "_attributes" in spec:
                self.write_attributes(grp, spec)

    # make sure specified path exists in group. if not, create it
    def ensure_path(self, grp, path):
        subs = path.split('/')
        for i in range(len(subs)):
            if len(subs[i]) == 0:
                continue
            if subs[i] not in grp:
                grp = grp.create_group(subs[i])
            else:
                grp = grp[subs[i]]

    def write_dataset_as_softlink(self, grp, path, field, spec):
        self.ensure_path(grp, path)
        # create external link for this field
        file_path = spec["_value_softlink_file"]
        dataset_path = spec["_value_softlink"]
        # create external link
        if len(path) > 0:
            grp = grp[path]
        grp[field] = h5py.ExternalLink(file_path, dataset_path)

    def write_dataset_as_hardlink(self, grp, path, field, spec):
        self.ensure_path(grp, path)
        # create hard link for this field
        # kernel will manage documenting hard links, after all 
        #   are created
        dataset_path = spec["_value_hardlink"] 
        #dataset_path = spec["_value_hardlink"] + "/" + field
        if len(path) > 0:
            grp = grp[path]
        if not dataset_path.startswith('/'):
            dataset_path = "/" + dataset_path
        #print("-------------")
        #print("Writing link:")
        #print("\tgrp: '%s'" % grp)
        #print("\tpath: '%s'" % path)
        #print("\tfield: '%s'" % field)
        #print("\tfullpath: '%s'" % dataset_path)
        grp[field] = h5py.SoftLink(dataset_path)
        #grp[field] = self.file_pointer[dataset_path]

    def write_dataset_to_file(self, grp, path, field, spec):
        self.ensure_path(grp, path)
        # advance group to specified location in path
        if len(path) > 0:
            grp = grp[path]
        # make sure dataset (or group) w/ this name doesn't exist already
        if field in grp:
            self.fatal_error("Field %s already exists" % field)
        # data not from link -- create dataset and set data
        varg = {}
        varg["name"] = field
        # get dtype
        if "_datatype" in spec:
            if spec["_datatype"] == "unrestricted":
                val = spec["_value"]
                # convert unicode to string
                # not internationalization-friendly, but 
                #   makes first version easier
                if isinstance(val, str):
                    varg["dtype"] = 'str'
                    spec["_value"] = str(spec["_value"])
                elif isinstance(val, str):
                    varg["dtype"] = 'str'
            else:
                varg["dtype"] = spec["_datatype"]
        elif isinstance(varg["data"], str):
            # string-handling logic below requires string dtype be labeled
            varg["dtype"] = 'str'
        elif isinstance(varg["data"], str):
            # string-handling logic below requires string dtype be labeled
            varg["dtype"] = 'str'
            spec["_value"] = str(spec["_value"])
        # create dataset
        # strings require special handling
        if "dtype" in varg and varg["dtype"] == 'str':
            # for now, assume that strings are simple or are
            #   stored in a 1D array
            value = spec["_value"]
            if type(value).__name__ == 'list':
                # make sure an empty list wasn't specified
                if len(value) == 0:
                    return
                # assume 1D array
                if isinstance(value[0], list):
                    self.fatal_error("Error -- writing multidimensional text arrays not yet supported (field %s)" % field)
                sz = -1 
                stype = ""
                for i in range(len(value)):
                    if sz < len(value[i]):
                        sz = len(value[i])
                        stype = "S%d" % (sz + 1)
                varg["shape"] = (len(value),)
                varg["dtype"] = stype
                # ignore compression/chunking for strings
                dset = grp.create_dataset(**varg)
                # space reserved for strings -- copy into place
                for i in range(len(value)):
                    dset[i] = np.string_(value[i])
            else:
                varg["data"] = np.string_(value)
                # don't specify dtype='str' -- h5py doesn't like that
                del varg["dtype"]
                ## ignore compression/chunking request for strings
                #dset = grp.create_dataset(**varg)
                if self.auto_compress:
                    varg["compression"] = 4
                    varg["chunks"] = True
                    try:
                        # try to use compression -- if we get a type error,
                        #   disable and try again
                        dset = grp.create_dataset(**varg)
                    except TypeError:
                        del varg["compression"]
                        del varg["chunks"]
                        dset = grp.create_dataset(**varg)
                else:
                    dset = grp.create_dataset(**varg)
        else:
            # try to use compression -- if we get a type error, disable
            #   and try again
            varg["data"] = spec["_value"]
            if self.auto_compress:
                varg["compression"] = 4
                varg["chunks"] = True
                try:
                    dset = grp.create_dataset(**varg)
                except TypeError:
                    del varg["compression"]
                    del varg["chunks"]
                    try:
                        dset = grp.create_dataset(**varg)
                    except Exception as e:
                        print("Exception text: %s" % str(e))
                        print("** Internal error **")
                        print("Dataset: %s" + str(varg["name"]))
                        print("Data follows.")
                        print(varg["data"])
                        raise
            else:
                dset = grp.create_dataset(**varg)
        if "_attributes" in spec:
            for k in spec["_attributes"]:
                if k.startswith('_'):
                    continue    # internal field -- nothing to write out
                if k == "<>" or k == "[]":
                    continue    # template
                # make a shorthand description of dictionary block
                block = spec["_attributes"][k]
                if "_value" in block:
                    val = block["_value"]
                    if "_datatype" in block:
                        valatt = block["_datatype"]
                        if valatt == "str":
                            val = np.string_(val)
                    try:
                        dset.attrs[k] = val
                    except RuntimeError as re:
                        print("Error storing attribute for field '%s'" % field)
                        print(re)
                        print("Value is: ")
                        print("data size is %d" % (len(val)))
                        print(val)
                        raise


