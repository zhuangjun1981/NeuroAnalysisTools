import os
import NeuroAnalysisTools.PreprocessPipeline as pp

def run():
    date = '200625'
    mid = '513381'
    scope = 'sutter'

    identifier = 'zstack_zoom2'

    zstack_chs = ('green', 'red')
    frame_per_step = 50
    ch_rf = 'red'
    chs_apply = ('green', 'red')
    process_num = 4

    data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data"

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

    ppr.reorganize_unaveraged_zstack_files(data_folder=os.path.join(sess_folder, identifier),
                                           identifier=identifier, channels=zstack_chs,
                                           frames_per_step=frame_per_step)

    ppr.motion_correction_unaveraged_zstack(data_folder=os.path.join(sess_folder, identifier, identifier),
                                            reference_channel_name=ch_rf, process_num=process_num)

    ppr.apply_offsets_unaveraged_zstack(data_folder=os.path.join(sess_folder, identifier, identifier),
                                        reference_channel_name=ch_rf,
                                        apply_channel_names=chs_apply,
                                        process_num=process_num)

    ppr.get_mean_projection_unaveraged_zstack(data_folder=os.path.join(sess_folder, identifier, identifier),
                                              save_folder=curr_folder, channels=chs_apply)

    ppr.motion_correction_zstack_all_steps(data_folder=curr_folder, save_folder=curr_folder,
                                           identifier=identifier,
                                           reference_channel_name=ch_rf,
                                           apply_channel_names=chs_apply)

    ppr.transform_images(data_folder=curr_folder, save_folder=curr_folder,
                         identifiers=[identifier, 'corrected'],
                         scope=scope)

    ppr.save_projections_as_png_files(data_folder=curr_folder, save_folder=curr_folder,
                                      identifiers=[identifier, 'corrected', 'aligned'],
                                      save_prefix='{}-M{}'.format(date, mid),
                                      projection_type='max', saturation_level=5.)

    # ppr.remove_corrected_files_unaveraged_zstack(data_folder=os.path.join(sess_folder, identifier, identifier),
    #                                              channels=chs_apply, is_remove_img=True)
    #
    # ppr.remove_uncorrected_files_unaveraged_zstack(data_folder=os.path.join(sess_folder, identifier, identifier),
    #                                                channels=chs_apply)
    #
    # ppr.remove_all_tif_files(os.path.join(sess_folder, identifier))

if __name__ == "__main__":
    run()
