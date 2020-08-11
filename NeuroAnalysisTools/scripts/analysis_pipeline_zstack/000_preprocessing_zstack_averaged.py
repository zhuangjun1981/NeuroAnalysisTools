import os
import NeuroAnalysisTools.PreprocessPipeline as pp

date = '200227'
mid = '500578'
scope = 'sutter'

vasmap_2p_chs = ('green', 'red')
zstack_chs = ('green', 'red')
ch_rf = 'red'
chs_apply = ('green', 'red')

# windows
data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data"

# # ubuntu
# data_folder = '/media/nc-ophys/jun/raw_data'

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

ppr = pp.Preprocessor()

if scope == 'sutter':
    folder_suffix = '2p'
elif scope == 'deepscope' or scope == 'DeepScope':
    folder_suffix = 'deepscope'
else:
    raise LookupError('Do not understand scope type.')

sess_folder = os.path.join(data_folder, '{}-M{}-{}'.format(date, mid, folder_suffix))

# ppr.get_vasmap_wf(data_folder=os.path.join(sess_folder, 'vasmap_wf_green'),
#                   save_folder=curr_folder,
#                   scope=scope, identifier='JCamF',
#                   save_suffix='_green')
#
# ppr.save_png_vasmap(tif_path='vasmap_wf_rotated_green.tif',
#                     prefix='{}-M{}'.format(date, mid),
#                     saturation_level=5.)
#
# ppr.get_vasmap_wf(data_folder=os.path.join(sess_folder, 'vasmap_wf_red'),
#                   save_folder=curr_folder,
#                   scope=scope, identifier='JCamF',
#                   save_suffix='_red')
#
# ppr.save_png_vasmap(tif_path='vasmap_wf_rotated_red.tif',
#                     prefix='{}-M{}'.format(date, mid),
#                     saturation_level=5.)
#
# ppr.get_vasmap_2p(data_folder=sess_folder, save_folder=curr_folder,
#                   scope=scope, channels=vasmap_2p_chs, identifier='vasmap_2p',
#                   is_equalize=False)
#
# for vmch in vasmap_2p_chs:
#     ppr.save_png_vasmap(tif_path='vasmap_2p_{}_rotated.tif'.format(vmch),
#                         prefix='{}-M{}'.format(date, mid), saturation_level=10.)

ppr.split_channels(data_folder=sess_folder, save_folder=curr_folder,
                   identifier='zstack_zoom1', channels=zstack_chs)

ppr.split_channels(data_folder=sess_folder, save_folder=curr_folder,
                   identifier='zstack_zoom2', channels=zstack_chs)

ppr.motion_correction_zstack(data_folder=curr_folder, save_folder=curr_folder,
                             identifier='zstack_zoom1',
                             reference_channel_name=ch_rf,
                             apply_channel_names=chs_apply)

ppr.motion_correction_zstack(data_folder=curr_folder, save_folder=curr_folder,
                             identifier='zstack_zoom2',
                             reference_channel_name=ch_rf,
                             apply_channel_names=chs_apply)

ppr.transform_images(data_folder=curr_folder, save_folder=curr_folder,
                     identifiers=['zstack', 'corrected'],
                     scope=scope)

ppr.save_projections_as_png_files(data_folder=curr_folder, save_folder=curr_folder,
                                  identifiers=['zstack', 'corrected', 'aligned'],
                                  save_prefix='{}-M{}'.format(date, mid),
                                  projection_type='max', saturation_level=5.)


