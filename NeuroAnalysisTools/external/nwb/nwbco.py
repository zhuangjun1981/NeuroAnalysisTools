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

# metadata storage constants
# these are so users don't have to remember exact path to
#   meatadata fields, and to avoid/detect typos
# all constants here are for storing data in the files top-level
#   group 'general'
#
# some are string constants and others are functions that
#   return strings. functions are used when there's a variable
#   element in metadata storage path

DATA_COLLECTION = "data_collection"
EXPERIMENT_DESCRIPTION = "experiment_description"
EXPERIMENTER = "experimenter"
INSTITUTION = "institution"
LAB = "lab"
NOTES = "notes"
PROTOCOL = "protocol"
PHARMACOLOGY = "pharmacology"
RELATED_PUBLICATIONS = "related_publications"
SESSION_ID = "session_id"
SLICES = "slices"
STIMULUS = "stimulus"
SURGERY = "surgery"
VIRUS = "virus"

SUBJECT = "subject/description"
SUBJECT_ID = "subject/subject_id"
SPECIES = "subject/species"
GENOTYPE = "subject/genotype"
SEX = "subject/sex"
AGE = "subject/age"
WEIGHT = "subject/weight"

def DEVICE(name):
    return "devices/" + name

########################################################################
# for extra-cellular ephys recordings

EXTRA_ELECTRODE_MAP = "extracellular_ephys/electrode_map"
EXTRA_ELECTRODE_GROUP = "extracellular_ephys/electrode_group"
EXTRA_IMPEDANCE = "extracellular_ephys/impedance"
EXTRA_FILTERING = "extracellular_ephys/filtering"

def EXTRA_CUSTOM(name):
    return "extracellular_ephys/" + name
def EXTRA_SHANK_DESCRIPTION(shank):
    return "extracellular_ephys/" + shank + "/description"
def EXTRA_SHANK_LOCATION(shank):
    return "extracellular_ephys/" + shank + "/location"
def EXTRA_SHANK_DEVICE(shank):
    return "extracellular_ephys/" + shank + "/device"
def EXTRA_SHANK_CUSTOM(shank, name):
    return "extracellular_ephys/" + shank + "/" + name

########################################################################
# for intra-cellular ephys recordings

def INTRA_CUSTOM(name):
    return "intracellular_ephys/" + name
def INTRA_ELECTRODE_DESCRIPTION(electrode):
    return "intracellular_ephys/" + electrode + "/description"
def INTRA_ELECTRODE_FILTERING(electrode):
    return "intracellular_ephys/" + electrode + "/filtering"
def INTRA_ELECTRODE_DEVICE(electrode):
    return "intracellular_ephys/" + electrode + "/device"
def INTRA_ELECTRODE_LOCATION(electrode):
    return "intracellular_ephys/" + electrode + "/location"
def INTRA_ELECTRODE_RESISTANCE(electrode):
    return "intracellular_ephys/" + electrode + "/resistance"
def INTRA_ELECTRODE_SLICE(electrode):
    return "intracellular_ephys/" + electrode + "/slice"
def INTRA_ELECTRODE_SEAL(electrode):
    return "intracellular_ephys/" + electrode + "/seal"
def INTRA_ELECTRODE_INIT_ACCESS_RESISTANCE(electrode):
    return "intracellular_ephys/" + electrode + "/initial_access_resistance"
def INTRA_ELECTRODE_CUSTOM(electrode, name):
    return "intracellular_ephys/" + electrode + "/" + name

########################################################################
# for optophysiology imaging

def IMAGE_CUSTOM(name):
    return "optophysiology/" + name
def IMAGE_SITE_DESCRIPTION(site):
    return "optophysiology/" + site + "/description"
def IMAGE_SITE_MANIFOLD(site):
    return "optophysiology/" + site + "/manifold"
def IMAGE_SITE_REFERENCE_FRAME(site):
    return "optophysiology/" + site + "/reference_frame"
def IMAGE_SITE_INDICATOR(site):
    return "optophysiology/" + site + "/indicator"
def IMAGE_SITE_EXCITATION_LAMBDA(site):
    return "optophysiology/" + site + "/excitation_lambda"
def IMAGE_SITE_CHANNEL_LAMBDA(site, channel):
    return "optophysiology/" + site + "/" + channel + "/emission_lambda"
def IMAGE_SITE_CHANNEL_DESCRIPTION(site, channel):
    return "optophysiology/" + site + "/" + channel + "/description"
def IMAGE_SITE_IMAGING_RATE(site):
    return "optophysiology/" + site + "/imaging_rate"
def IMAGE_SITE_LOCATION(site):
    return "optophysiology/" + site + "/location"
def IMAGE_SITE_DEVICE(site):
    return "optophysiology/" + site + "/device"
def IMAGE_SITE_CUSTOM(site, name):
    return "optophysiology/" + site + "/" + name

########################################################################
# for optogentics

def OPTOGEN_CUSTOM(name):
    return "optogenetics/" + name
def OPTOGEN_SITE_DESCRIPTION(site):
    return "optogenetics/" + site + "/description"
def OPTOGEN_SITE_DEVICE(site):
    return "optogenetics/" + site + "/device"
def OPTOGEN_SITE_LAMBDA(site):
    return "optogenetics/" + site + "/lambda"
def OPTOGEN_SITE_LOCATION(site):
    return "optogenetics/" + site + "/location"
def OPTOGEN_SITE_CUSTOM(site, name):
    return "optogenetics/" + site + "/" + name



