import os
import NeuroAnalysisTools.NwbTools as nt

date_recorded = '190503'
mouse_id = '439939'
sess_num = '110'

experimenter = 'Jun'
genotype = 'Vipr2-IRES2-Cre-neo'
sex = 'Male'
age = '147'
indicator = 'GCaMP6s'
imaging_rate = 30. # deepscope 37.
imaging_depth = '150 microns' # deepscope [150, 100, 50] or [300, 250, 200]
imaging_location = 'visual cortex'
imaging_device = 'Sutter' # or DeepScope
imaging_excitation_lambda = '920 nanometers' # deepscope 940 nanometers

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

notebook_path = os.path.join(curr_folder, 'notebook.txt')
with open(notebook_path, 'r') as ff:
    notes = ff.read()

general = nt.DEFAULT_GENERAL
general['experimenter'] = experimenter
general['subject']['subject_id'] = mouse_id
general['subject']['genotype'] = genotype
general['subject']['sex'] = sex
general['subject']['age'] = age
general['optophysiology'].update({'imaging_plane_0': {}})
general['optophysiology']['imaging_plane_0'].update({'indicator': indicator})
general['optophysiology']['imaging_plane_0'].update({'imaging_rate': imaging_rate})
general['optophysiology']['imaging_plane_0'].update({'imaging_depth': imaging_depth})
general['optophysiology']['imaging_plane_0'].update({'location': imaging_location})
general['optophysiology']['imaging_plane_0'].update({'device': imaging_device})
general['optophysiology']['imaging_plane_0'].update({'excitation_lambda': imaging_excitation_lambda})
general['notes'] = notes

file_name = date_recorded + '_M' + mouse_id + '_' + sess_num + '.nwb'

rf = nt.RecordedFile(os.path.join(curr_folder, file_name), identifier=file_name[:-4], description='')
rf.add_general(general=general)
rf.close()



