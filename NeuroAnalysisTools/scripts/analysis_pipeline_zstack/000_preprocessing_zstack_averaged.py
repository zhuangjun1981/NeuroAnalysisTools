import os
import NeuroAnalysisTools.PreprocessPipeline as pp

date = '200715'
mid = '533920'
scope = 'sutter'

identifier = 'zstack_zoom2'
vasmap_2p_chs = ('green',)
zstack_chs = ('green',)
ch_rf = 'green'
chs_apply = ('green',)

source_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data"

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

if scope == 'sutter':
    sess_folder = os.path.join(source_folder, '{}-M{}-2p'.format(date, mid))
    vasmap_wf_identifier = 'JCam'
elif scope == 'deepscope':
    sess_folder = os.path.join(source_folder, '{}-M{}-deepscope'.format(date, mid))
    vasmap_wf_identifier = 'vasmap_wf'
else:
    raise LookupError('Do not understand scope type.')

ppr = pp.Preprocessor()

# # get 2p vasculature maps as tif
# _ = ppr.get_vasmap_2p(data_folder=os.path.join(sess_folder, 'vasmap_2p'),
#                       scope=scope,
#                       channels=vasmap_2p_chs,
#                       identifier='vasmap_2p',
#                       is_equalize=False,
#                       save_folder=curr_folder)
#
# # save aligned 2p vasculature maps as png
# for vasmap_2p_ch in vasmap_2p_chs:
#     ppr.save_png_vasmap(tif_path=os.path.join(curr_folder, 'vasmap_2p_{}_rotated.tif'.format(vasmap_2p_ch)),
#                         prefix='{}_{}'.format(date, mid), saturation_level=10)
#
# # get widefield vasculature maps as tif
# _ = ppr.get_vasmap_wf(data_folder=os.path.join(sess_folder, 'vasmap_wf'),
#                       save_folder=curr_folder,
#                       scope=scope,
#                       identifier=vasmap_wf_identifier)
#
# # save aligned widefield vasculature maps as png
# ppr.save_png_vasmap(tif_path=os.path.join(curr_folder, 'vasmap_wf_rotated.tif'),
#                     prefix='{}_{}'.format(date, mid), saturation_level=5)
#
#
# ppr.split_channels(data_folder=os.path.join(sess_folder, identifier),
#                    save_folder=curr_folder,
#                    identifier=identifier, channels=zstack_chs)
#
# ppr.motion_correction_zstack_all_steps(data_folder=curr_folder, save_folder=curr_folder,
#                                        identifier=identifier,
#                                        reference_channel_name=ch_rf,
#                                        apply_channel_names=chs_apply)
#
# ppr.transform_images(data_folder=curr_folder, save_folder=curr_folder,
#                      identifiers=[identifier, 'corrected'],
#                      scope=scope)
#
# ppr.save_projections_as_png_files(data_folder=curr_folder, save_folder=curr_folder,
#                                   identifiers=[identifier, 'corrected', 'aligned'],
#                                   save_prefix='{}-M{}'.format(date, mid),
#                                   projection_type='max', saturation_level=5.)


