    
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from nilearn import plotting
from nimare import utils

def select_subject(subject_idx) : 
    """ 
    Select a subject from the dataset of the second experiment.
    Parameters
    ----------
    subject_idx : int
        Index of the subject to select.
    Returns
    -------
    selected_subject : dict
        Dictionary containing the selected subject's data, number and index.
    """    
    selected_subject_data = alldata_exp2[subject_idx] #alldata_exp2 is a global variable in the notebook, containing the dataset from the second experiment
    selected_subject = {'selected_subject_data' : selected_subject_data,
                        'selected_subject_number' : subject_idx+1,
                        'selected_subject_idx' : subject_idx,
            }
    return selected_subject

def describe_one_subject_data(selected_idx):
    """
    Describe the data of a selected subject.
    Parameters
    ----------
    selected_idx : int
        Index of the subject to describe.
    Returns
    -------
    None
    Prints the type, shape, and first values of each key in the subject's data.
    """
    selected_subject = select_subject(selected_idx)
    selected_subject_data = selected_subject['selected_subject_data']
    for k, v in selected_subject_data.items():
        if isinstance(v, np.ndarray):
            info = f"{k:20}: type={str(type(v)):30} shape = {str(np.shape(v)):15}"
            if np.ndim(v) == 1 or (np.ndim(v) == 2 and v.shape[1] == 1):
                v = v.reshape(-1)
                print(f"{info} First values = {v[:5]}")
            else:
                print(info)
        elif isinstance(v, list):
            print(f"{k:20}: type={str(type(v)):30} len = {str(len(v)):15}   First values = {v[:3]}")
        else:
            try:
                length = len(v)
            except TypeError:
                length = "N/A"
            print(f"{k:20}: type={str(type(v)):30} len = {str(length):15}   {v}")

def show_electrodes_mapping(selected_subject) :
    """ 
    Visualize the electrode locations on a 3D brain model and create a DataFrame with electrode metadata.
    Parameters
    ----------
    selected_subject : dict
        Dictionary containing the selected subject's data.
    Returns
    -------
    view : nilearn.plotting.view_connectome
        Interactive 3D visualization of electrode locations.
    electrodes_df : pandas.DataFrame
        DataFrame containing metadata for each electrode.
    """

    selected_subject_data = selected_subject['selected_subject_data']
    data_electrodes = {
        'electrode_idx' : range(len(selected_subject_data['hemisphere'])),
        'hemisphere' : selected_subject_data['hemisphere'],
        'lobe' : selected_subject_data['lobe'],
        'gyrus' : selected_subject_data['gyrus'],
        'Brodmann_Area' : selected_subject_data['Brodmann_Area']
        }
    electrodes_df = pd.DataFrame(data_electrodes)
    
    plt.figure(figsize=(8, 8))
    locs = selected_subject_data['locs']
    view = plotting.view_markers(utils.tal2mni(locs),
                                marker_labels=['%d'%k for k in np.arange(locs.shape[0])],
                                marker_color='purple',
                                marker_size=5)
    return view, electrodes_df
    

def data_ext(selected_subject) :
    """
    Extract and process key event timelines from the selected subject's data.
    Parameters
    ----------
    selected_subject : dict
        Dictionary containing the selected subject's data.
    Returns
    -------
    imageonset_timesteps_0_1 : np.ndarray
        Binary array of all timesteps and indicating image onset times.
    imageoffset_timesteps_0_1 : np.ndarray
        Binary array of all timesteps and indicating image offset times.
    keypress_timesteps_0_1 : np.ndarray
        Binary array of all timesteps and indicating keypress times.
    all_timesteps : np.ndarray
        Array of all timesteps.
    nb_all_timesteps : int
        Total number of timesteps.
    exposedimage_number_timeline : np.ndarray
        Array indicating the image number displayed at each timestep.
    images_recognized_asface_0_1_630 : np.ndarray
        Binary array indicating images recognized as faces.
    interval_keypress_630 : np.ndarray
        Array of intervals between image onset and keypress for images recognized as faces.
    """

    selected_subject_data = selected_subject['selected_subject_data']
    imageonset_timesteps_630 = selected_subject_data['t_on']
    imageoffset_timesteps_630 = selected_subject_data['t_off']
    all_timesteps = np.arange(1, selected_subject_data['V'].shape[0]+1, dtype = int)
    nb_all_timesteps = selected_subject_data['V'].shape[0]
    keypress_timesteps = selected_subject_data['key_press']

    imageonset_timesteps_0_1 = np.zeros(nb_all_timesteps, dtype=int)
    imageonset_timesteps_0_1[imageonset_timesteps_630-1] = 1
    imageoffset_timesteps_0_1 = np.zeros(nb_all_timesteps, dtype=int)
    imageoffset_timesteps_0_1[imageoffset_timesteps_630-1] = 1
    keypress_timesteps_0_1 = np.zeros(nb_all_timesteps, dtype=int)
    keypress_timesteps_0_1[keypress_timesteps-1] = 1

    nb_images_recognized_asface = keypress_timesteps.size
    index_images_recognized_asface = np.zeros(nb_images_recognized_asface, dtype=int)
    interval_to_keypress_when_recognized_asface = np.zeros(nb_images_recognized_asface)

    for i in np.arange(nb_images_recognized_asface) : # For each image recongnized as face by the subject, store its index + interval de press the key
        index = imageonset_timesteps_630[imageonset_timesteps_630 < keypress_timesteps[i]].size-1
        interval = keypress_timesteps[i] - imageonset_timesteps_630[index]

        if interval < 200 and index > 0:  # If the keypress is too fast (within 200 ms after image onset), it is assumed that the subject was reacting to the previous image
            index -= 1
            interval = keypress_timesteps[i] - imageonset_timesteps_630[index]

        index_images_recognized_asface[i] = index
        interval_to_keypress_when_recognized_asface[i] = interval

    images_recognized_asface_0_1_630 = np.zeros_like(imageonset_timesteps_630)
    images_recognized_asface_0_1_630[index_images_recognized_asface] = 1
    interval_keypress_630 = np.full_like(imageonset_timesteps_630, np.nan, dtype=float)
    interval_keypress_630[index_images_recognized_asface] = interval_to_keypress_when_recognized_asface

    exposedimage_number_timeline = np.full(nb_all_timesteps, np.nan, dtype=float)
    for i in np.arange(imageonset_timesteps_630.size) :
        exposedimage_number_timeline[imageonset_timesteps_630[i]-1 : imageoffset_timesteps_630[i]-1] = i+1   # stores the image number, not its index
    
    return imageonset_timesteps_0_1, imageoffset_timesteps_0_1, keypress_timesteps_0_1, all_timesteps, nb_all_timesteps, exposedimage_number_timeline, images_recognized_asface_0_1_630, interval_keypress_630

def construct_images_df(selected_subject) :
    """
    Construct a DataFrame containing metadata and behavioral data for each image presented to the selected subject.
    Parameters
    ----------
    selected_subject : dict
        Dictionary containing the selected subject's data.
    Returns
    -------
    selected_subject_images_df : pandas.DataFrame
        DataFrame containing metadata and behavioral data for each image.
    """

    selected_subject_data = selected_subject['selected_subject_data']
    _, _, _, _, _, _, images_recognized_asface_0_1_630, interval_keypress_630 = data_ext(selected_subject)
    selected_subject_images_df = pd.DataFrame(data = {
        'Image_ID' : selected_subject_data['stim_id'],
        'Category' : np.where(selected_subject_data['stim_cat'] == 1, 'house', 'face').reshape(-1),
        'Noise' : selected_subject_data['stim_noise'].reshape(-1),
        'Image_onset' : selected_subject_data['t_on'],
        'Image_offset' : selected_subject_data['t_off'],
        'Display_duration' : selected_subject_data['t_off']-selected_subject_data['t_on'],
        'Key_pressed' : images_recognized_asface_0_1_630,
        'Interval_keypress' : interval_keypress_630
    })
    
    return selected_subject_images_df

def get_noise_levels(data_exp2) :
    """
    Extract and sort unique noise levels from the dataset.
    Parameters
    ----------
    data_exp2 : np.ndarray
        Array of dictionaries containing data for all subjects in the second experiment.
    Returns
    -------
    sorted_noise_levels : np.ndarray
        Array of unique noise levels in ascending order.
    sorted_noise_levels_labels : list
        List of string labels for the noise levels in ascending order (e.g., '0%', '5%', etc.).
    n_noise_levels : int
        Number of unique noise levels.
    """
    noise_levels = np.unique(data_exp2[0]['stim_noise'])
    sorted_noise_levels = np.sort(noise_levels)
    sorted_noise_levels_labels = [f'{sorted_noise_levels[i]}%' for i in range(sorted_noise_levels.size)]
    n_noise_levels = len(sorted_noise_levels)
    return sorted_noise_levels, sorted_noise_levels_labels, n_noise_levels 

def get_TPNF(selected_subject_images_df, args, per_noise_level=False) :
    """
    Calculate True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN) for a selected subject.
    Parameters
    ----------
    selected_subject_images_df : pandas.DataFrame
        DataFrame containing metadata and behavioral data for each image for the selected subject.
    args : dict
        Dictionary containing additional arguments, including 'sorted_noise_levels'.
    per_noise_level : bool, optional
        If True, calculate TP, TN, FP, and FN for each noise level separately. 
        If False, calculate overall values regardless of noise level.
        Default is False.
    Returns
    -------
    TP : int or np.ndarray
        Number of TP, or array of TP per noise level.
    TN : int or np.ndarray
        Number of TN, or array of TN per noise level.
    FP : int or np.ndarray
        Number of FP, or array of FP per noise level.
    FN : int or np.ndarray
        Number of FN, or array of FN per noise level.
    """
    images_df = selected_subject_images_df
    if per_noise_level == False :
        TP = images_df[(images_df['Category'] == 'face') & (images_df['Key_pressed'] == 1)].shape[0] # Keypress when face stimulus
        TN = images_df[(images_df['Category'] == 'house') & (images_df['Key_pressed'] == 0)].shape[0] # No keypress when house stimulus
        FP = images_df[(images_df['Category'] == 'house') & (images_df['Key_pressed'] == 1)].shape[0] # Keypress when house stimulus
        FN = images_df[(images_df['Category'] == 'face') & (images_df['Key_pressed'] == 0)].shape[0] # No keypress when face stimulus 
    else :
        sorted_noise_levels = args['sorted_noise_levels']
        TP = np.zeros_like(sorted_noise_levels)
        TN = np.zeros_like(sorted_noise_levels)
        FP = np.zeros_like(sorted_noise_levels)
        FN = np.zeros_like(sorted_noise_levels)
        for i, level in enumerate(sorted_noise_levels) :
            TP_l = images_df[(images_df['Category'] == 'face') & (images_df['Key_pressed'] == 1) & (images_df['Noise'] == level)].shape[0] 
            TN_l = images_df[(images_df['Category'] == 'house') & (images_df['Key_pressed'] == 0) & (images_df['Noise'] == level)].shape[0] 
            FP_l = images_df[(images_df['Category'] == 'house') & (images_df['Key_pressed'] == 1) & (images_df['Noise'] == level)].shape[0] 
            FN_l = images_df[(images_df['Category'] == 'face') & (images_df['Key_pressed'] == 0) & (images_df['Noise'] == level)].shape[0] 
            TP[i] = TP_l
            TN[i] = TN_l
            FP[i] = FP_l
            FN[i] = FN_l
    return TP, TN, FP, FN

def get_accuracy(subjects_idx:list, args, per_noise_level=False) :
    """
    Calculate accuracy of keypress responses for a list of subjects.
    Parameters
    ----------
    subjects_idx : list
        List of subject indices to calculate accuracy for.
    args : dict
        Dictionary containing additional arguments, including 'sorted_noise_levels'.
    per_noise_level : bool, optional
        If True, calculate accuracy for each noise level separately.
        If False, calculate overall accuracy regardless of noise level.
        Default is False.
    Returns
    -------
    acc_all : np.ndarray
        1D array of accuracy values for each subject, or a 2D array of accuracy values per noise level for each subject.
    """
    acc_all = []
    for subj_idx in subjects_idx :
        selected_subject = select_subject(subj_idx)
        images_df = construct_images_df(selected_subject)
        if per_noise_level==False :
            TP, TN, FP, FN = get_TPNF(images_df, args, per_noise_level)
            accuracy = (TP+TN)/(TP+TN+FP+FN)
            acc_all.append(accuracy)
        else :
            sorted_noise_levels = args['sorted_noise_levels']
            TP, TN, FP, FN = get_TPNF(images_df, args, per_noise_level=True)
            assert isinstance(TP, np.ndarray)
            assert isinstance(TN, np.ndarray)
            assert isinstance(FP, np.ndarray)
            assert isinstance(FN, np.ndarray)
            acc_per_noise_level_subj = []
            for j in np.arange(sorted_noise_levels.size) :
                accuracy = (TP[j]+TN[j])/(TP[j]+TN[j]+FP[j]+FN[j])
                acc_per_noise_level_subj.append(accuracy)
            acc_all.append(acc_per_noise_level_subj)
    return np.array(acc_all)

def get_face_recognition_rate(subjects_idx:list, args, per_noise_level=False) :   
    """
    Calculate rate of correctly identified faces for a list of subjects.
    Parameters
    ----------
    subjects_idx : list
        List of subject indices to calculate face recognition rate for.
    args : dict
        Dictionary containing additional arguments, including 'sorted_noise_levels'.
    per_noise_level : bool, optional
        If True, calculate face recognition rate for each noise level separately.
        If False, calculate overall face recognition rate regardless of noise level.
        Default is False.
    Returns
    -------
    rate_all : np.ndarray
        1D array of face recognition rates for each subject, or a 2D array of face recognition rates per noise level for each subject.
    """ 
    rate_all = []
    for subj_idx in subjects_idx :
        selected_subject = select_subject(subj_idx)
        images_df = construct_images_df(selected_subject)
        if per_noise_level==False :
            n_identified_faces = images_df[(images_df['Category'] == 'face') & (images_df['Key_pressed'] == 1)].shape[0]
            n_all_faces = images_df[(images_df['Category'] == 'face')].shape[0]
            rate = n_identified_faces/n_all_faces
            rate_all.append(rate)
        else :
            sorted_noise_levels = args['sorted_noise_levels']
            rate_per_noise_level_subj = []
            for level in sorted_noise_levels :
                n_identified_faces = images_df[(images_df['Category'] == 'face') & (images_df['Key_pressed'] == 1) & (images_df['Noise'] == level)].shape[0]
                n_all_faces = images_df[(images_df['Category'] == 'face') & (images_df['Noise'] == level)].shape[0]
                rate_per_noise_level_subj.append(n_identified_faces/n_all_faces)
            rate_all.append(rate_per_noise_level_subj)
    return np.array(rate_all)

def fit_plot_sigmoid_curve(subjects_idx:list, metrics, args, figsize=(10,5), modif=False) :
    """
    Fit and plot a sigmoid curve to the specified metric (face recognition rate or accuracy) across noise levels for a list of subjects.
    Parameters
    ----------
    subjects_idx : list
        List of subject indices to plot.
    metrics : str
        Metric to plot: 'face_recognition_rate' or 'accuracy'.
    args : dict
        Dictionary containing additional arguments, including arguments about noise levels and subjects.
    figsize : tuple, optional
        Size of the figure to create. Default is (10, 5).
    modif : bool, optional
        If True, allows upcoming modifications to the plot without displaying it immediately.
    Returns
    -------
    None
    Displays the plot if modif is False, otherwise allows for further custom modifications to the plot before calling plt.show().
    """
    assert metrics in ('face_recognition_rate', 'accuracy')
    key_subjects_index = args['key_subjects_index']
    key_subjects_colors = args['key_subjects_colors']
    sorted_noise_levels = args['sorted_noise_levels']
    sorted_noise_levels_labels = args['sorted_noise_levels_labels']
    
    def sigmoid_function(x, L, x0, k, b):
        return L / (1 + np.exp(-k * (x - x0))) + b

    y_pred_all = []
    x_plot = np.linspace(0, 1, 300)
    plt.figure(figsize=figsize)

    if metrics == 'accuracy' :
        metrics_all = get_accuracy(subjects_idx, args, per_noise_level=True)
        plt.axhline(0.5, linestyle='--', color='gray', label='Chance level') #Chance level line
        plt.title("Keypress accuracy across noise levels", fontsize='x-large')
    else :
        metrics_all = get_face_recognition_rate(subjects_idx, args, per_noise_level=True)
        plt.title("Face recognition rate across noise levels", fontsize='x-large')

    for i, subj_idx in enumerate(subjects_idx) :
        x_observed = np.array(sorted_noise_levels)
        y_observed = metrics_all[i]
        popt, _ = curve_fit(sigmoid_function, x_observed/100.0, y_observed, p0=[1.0, 0.5, 10, 0.0])
        y_pred = sigmoid_function(x_plot, *popt)
        y_pred_all.append(y_pred)

        color = key_subjects_colors[key_subjects_index.index(subj_idx)]
        metric_rep = metrics.replace('_',' ')
        label = f"Observed {metric_rep} (Subject {subj_idx + 1})"
        plt.plot(x_observed, y_observed, 'o', color=color, label=label)
        plt.plot(x_plot * 100, y_pred, '-', color=color)

    plt.xlabel("Noise Level (%)", fontsize=14)
    plt.ylabel(f"{metric_rep.title()}", fontsize=14)
    plt.xticks(ticks=sorted_noise_levels, labels=sorted_noise_levels_labels)
    plt.yticks(ticks=np.arange(0, 1.01, 0.1))    
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(-0.02, 1.02)
    plt.tight_layout()
    if modif == False :
        plt.show()
        

def get_key_behavior(subjects_idx:list, args) :
    """
    Plot the number of keypresses across noise levels for a list of subjects.
    Parameters
    ----------
    subjects_idx : list
        List of subject indices to plot.
    args : dict
        Dictionary containing additional arguments, including arguments about noise levels and subjects.
    Returns
    -------
    None
    Displays a bar plot of the number of keypresses across noise levels for each subject.
    """
    nb_identified_faces = []
    key_subjects_colors = args['key_subjects_colors']
    colors = [key_subjects_colors[args['key_subjects_index'].index(idx)] for idx in subjects_idx]
    sorted_noise_levels = args['sorted_noise_levels']
    sorted_noise_levels_labels = args['sorted_noise_levels_labels']

    for subj_idx in subjects_idx :
        selected_subject = select_subject(subj_idx)
        images_df = construct_images_df(selected_subject)
        nb = np.zeros_like(sorted_noise_levels)
        for i, level in enumerate(sorted_noise_levels) :
            nb_l = images_df[(images_df['Key_pressed'] == 1) & (images_df['Noise'] == level)].shape[0]
            nb[i] = nb_l
        nb_identified_faces.append(nb)

    # plotting : 
    records = []
    for subj_idx, vals in zip(subjects_idx, nb_identified_faces):
        for noise, val in zip(sorted_noise_levels, vals):
            records.append({'Subject': f'S{subj_idx+1}', 'Noise': noise, 'Value': val})
    df = pd.DataFrame(records)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x='Noise', y='Value', hue='Subject', alpha=0.7, palette=colors)
    plt.ylabel("Number of keypresses", fontsize=14)
    plt.xlabel("Noise level", fontsize=14)
    plt.xticks(ticks=range(len(sorted_noise_levels)), labels=sorted_noise_levels_labels)
    plt.yticks(ticks=np.arange(0, np.max(np.concatenate(nb_identified_faces))+5, 5))
    plt.title("Keypress count across noise levels", fontsize='x-large')
    plt.tight_layout()
    plt.show()


def construct_timeline_events_df(selected_subject) :
    """
    Construct a DataFrame containing a full timeline of the second expermient, including discret events and electrode data.
    Parameters
    ----------
    selected_subject : dict
        Dictionary containing the selected subject's data.
    Returns
    -------
    timeline_df : pandas.DataFrame
        DataFrame containing the full timeline of the second experiment with events and electrode data.
    """
    selected_subject_data = selected_subject['selected_subject_data']
    imageonset_timesteps_0_1, imageoffset_timesteps_0_1, keypress_timesteps_0_1, all_timesteps, _, exposedimage_number_timeline, _, _ = data_ext(selected_subject)

    timeline_df = pd.DataFrame({
        'timeline': all_timesteps,
        'image_onset_0_1': imageonset_timesteps_0_1,
        'image_offset_0_1': imageoffset_timesteps_0_1,
        'keypress_0_1': keypress_timesteps_0_1,
        'displayed_image': exposedimage_number_timeline
    })

    for i in range (selected_subject_data['V'].shape[1]) :
        timeline_df[f"V_{i+1}"] = selected_subject_data['V'][:,i]

    return timeline_df


def plot_timeline(timeline_df, start_timestep, end_timestep, nb_segments, subject_idx):
    """
    Plot the timeline showing image onsets and keypress events between specified timesteps, divided into segments.
    Parameters
    ----------
    timeline_df : pandas.DataFrame
        DataFrame containing the full timeline of the second experiment with events and electrode data.
    start_timestep : int
        Starting timestep to consider for the plot.
    end_timestep : int
        Ending timestep to consider for the plot.
    nb_segments : int
        Number of segments to divide the specified timestep range into.
    subject_idx : int
        Index of the subject for labeling purposes.
    Returns
    -------
    None
    Displays the timeline plots for the specified timestep range and number of segments.
    """
    total_timesteps = end_timestep - start_timestep
    segment_size = total_timesteps // nb_segments

    for i in range(nb_segments):
        start = start_timestep + i * segment_size
        end = start_timestep + (i + 1) * segment_size if i < nb_segments - 1 else end_timestep

        plt.figure(figsize=(20, 2))
        plt.ylim((0, 0.2))
        plt.title(f"Segment {i+1}: timesteps {start}-{end - 1} of {int(timeline_df['timeline'].max())}) (Subject {subject_idx+1})")
        plt.eventplot(
            np.where(timeline_df['image_onset_0_1'][start:end] == 1)[0] + start,
            linelengths=0.2,
            linewidths=2,
            colors='black',
            lineoffsets=0
        )
        plt.eventplot(
            np.where(timeline_df['keypress_0_1'][start:end] == 1)[0] + start,
            linelengths=0.2,
            linewidths=2,
            colors='red',
            lineoffsets=0.2
        )
        plt.legend(['Image onset', 'Keypress'], loc='upper right')
        plt.xlabel("Time (ms)")
        plt.ylabel("Events")
        plt.tight_layout()
        plt.show()


def show_metadataframes(subject_idx, images_df=None, timeline_df=None, electrodes_df=None, view_electrodes=None):
    """
    Display metadata DataFrames for images, timeline events, and electrodes for a selected subject.
    Parameters
    ----------
    subject_idx : int
        Index of the subject for labeling purposes.
    images_df : pandas.DataFrame, optional
        DataFrame containing metadata for images. Default is None.
    timeline_df : pandas.DataFrame, optional
        DataFrame containing metadata for timeline events. Default is None.
    electrodes_df : pandas.DataFrame, optional
        DataFrame containing metadata for electrodes. Default is None.
    view_electrodes : nilearn.plotting.view_connectome, optional
        Interactive 3D visualization of electrode locations. Default is None.
    Returns
    -------
    None
    Displays the metadata DataFrames and electrode visualization if provided.
    """
    if images_df is not None:
        print(f"Images metadata (Subject {subject_idx+1}):")
        display(images_df.head())
        print("\n\n")
    if timeline_df is not None:
        print(f"Timeline events metadata (Subject {subject_idx+1}):")
        display(timeline_df.head())
        plot_timeline(timeline_df, nb_segments=1, start_timestep=0, end_timestep=70000, subject_idx=subject_idx)
        print("\n\n")
    if electrodes_df is not None:
        print(f"Electrodes metadata (Subject {subject_idx+1}):")
        print(f"Subject {subject_idx+1} has {electrodes_df.shape[0]} electrodes.")
        display(electrodes_df.head())
        display(view_electrodes)

def display_face_recognition_counts(selected_subject, args, grouped_noise_dict=None):
    """
    Display the number of faces correctly identified across noise levels for a selected subject.
    Parameters
    ----------
    selected_subject : dict
        Dictionary containing the selected subject's data.
    args : dict
        Dictionary containing additional arguments including arguments about noise levels. 
    grouped_noise_dict : dict, optional
        Dictionary defining grouped noise levels. If provided, counts will be aggregated by these groups.
    Returns
    -------
    None
    Displays a DataFrame with the number of faces correctly identified across noise levels or grouped noise levels.
    """
    images_df = construct_images_df(selected_subject)
    sorted_noise_levels = args['sorted_noise_levels']
    sorted_noise_levels_labels = args['sorted_noise_levels_labels']
    face_recognition_counts = (
        images_df[(images_df['Category'] == 'face') & (images_df['Key_pressed'] == 1)].groupby('Noise').size()
        .reindex(sorted_noise_levels, fill_value=0)
    )

    if grouped_noise_dict is None:   
        df = pd.DataFrame(data={
            'Noise Level': sorted_noise_levels_labels,
            'Faces correctly identified': face_recognition_counts.values
        }) 
        display(df.T.style.hide(axis='columns').set_caption(f"Number of faces correctly identified across noise levels (Subject {selected_subject['selected_subject_number']})"))
    
    else :
        grouped_counts = []
        for group in grouped_noise_dict.values():
            total = face_recognition_counts.loc[group].sum()
            grouped_counts.append(total)
        df = pd.DataFrame(data={
            'Noise Level': list(grouped_noise_dict.keys()),
            'Faces correctly identified': grouped_counts
        }) 
        pd.set_option('display.max_columns', None)
        display(df.T.style.hide(axis='columns').set_caption(
            f"Number of faces correctly identified by grouped noise levels (Subject {selected_subject['selected_subject_number']})"
        ))

def plot_interval_keypress_recog_faces(selected_subject, args):
    """
    Plot the keypress latency for images correctly identified as faces across noise levels for a selected subject.
    Parameters
    ----------
    selected_subject : dict
        Dictionary containing the selected subject's data.
    args : dict
        Dictionary containing additional arguments including arguments about noise levels.
    Returns
    -------
    None
    Displays a boxplot with jittered data points of keypress latency for images correctly identified as faces across noise levels.
    """
    images_df = construct_images_df(selected_subject)
    plt.figure(figsize=(10, 5))
    recog_faces_df = images_df[(images_df['Key_pressed'] == 1) & (images_df['Category'] == 'face')]
    sns.boxplot(x='Noise', y='Interval_keypress', data=recog_faces_df, color='orange', showfliers=False)
    sns.stripplot(x='Noise', y='Interval_keypress', data=recog_faces_df, jitter=True, alpha=0.5, size=4, color='black')
    plt.title(f"Keypress latency for images correctly identified as faces (Subject {selected_subject['selected_subject_number']})", fontsize='x-large')
    plt.xlabel("Noise level", fontsize='large')
    plt.ylabel("Keypress latency (ms)", fontsize='large')
    plt.xticks(ticks=range(len(args['sorted_noise_levels'])), labels=args['sorted_noise_levels_labels'], fontsize='medium')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def get_power_all_electrodes(selected_subject, freq_down, freq_up, smooth_power) :  
    """
    Compute the power of the neural signal for all electrodes for a selected subject, applying bandpass filtering and optional smoothing.
    Note:
    Raw voltage data is already z-scored and ambient line-noise rejected applying notch filters between 58-62, 118-120, and 178-182 Hz using 3rd-order Butterworth filters.    
    After rescaling the voltage to microvolts, a bandpass filter is applied between freq_down and freq_up (in Hz) using a 3rd-order Butterworth filter. 
    A Common Average Reference is then applied by subtracting the mean signal across all electrodes at each timestep. 
    The power is computed as the squared absolute value of the filtered signal. 
    If smooth_power is provided (in Hz), a low-pass filter is applied to the power signal using a 3rd-order Butterworth filter with the specified cutoff frequency.
    Finally, the baseline (periods with no image displayed) mean power is subtracted and the result is normalized by the baseline power.    
    
    Parameters
    ----------
    selected_subject : dict
        Dictionary containing the selected subject's data.
    freq_down : int
        Lower bound of the frequency band for bandpass filtering (in Hz).
    freq_up : int
        Upper bound of the frequency band for bandpass filtering (in Hz).
    smooth_power : int or None
        Cutoff frequency for low-pass filtering the power signal (in Hz). If None, no smoothing is applied.
    Returns
    -------
    power : np.ndarray
        2D array of power with shape (timesteps, electrodes).
    """
    selected_subject_data = selected_subject['selected_subject_data']
    timeline_df = construct_timeline_events_df(selected_subject)
    V_raw = selected_subject_data['V'].astype('float32')

    #Scaling uV :
    V_scaled = V_raw * selected_subject_data['scale_uv']

    #Filtring frequency band : 
    assert 0 < freq_down < freq_up, "Invalid frequencies"
    b, a = signal.butter(3, [freq_down, freq_up], btype='bandpass', fs=1000)
    V_freqfiltred = signal.filtfilt(b, a, V_scaled, axis=0)

    #Common Average Reference :
    V_freqfiltred = V_freqfiltred - V_freqfiltred.mean(axis=1, keepdims=True)

    #Get power :
    power = np.abs(V_freqfiltred)**2

    #Smoothing power :
    if smooth_power is not None :
        b, a = signal.butter(3, smooth_power, btype='low', fs=1000)
        power = signal.filtfilt(b, a, power, 0)

    # Normalizing by baseline (periods with no image displayed) :
    baseline_mask = timeline_df['displayed_image'].isna()
    baseline_mean = power[baseline_mask].mean(axis=0)    
    power = (power - baseline_mean) / baseline_mean
    
    return power

def make_noise_combinations_dict(groupings_list):
    """
    Create a dictionary of noise level combinations based on provided groupings.
    Parameters
    ----------
    groupings_list : list of lists
        List containing sublists of noise levels to be grouped together.
    Returns
    -------
    noise_dict : dict
        Dictionary where keys are string labels of noise level combinations and values are lists of the corresponding noise levels.
    """
    used_levels = set()
    noise_dict = {}

    max_value = max(max(sublist) for sublist in groupings_list)
    available_levels_list = list(range(0,max_value+1,5))

    for group in groupings_list:
        unique = [level for level in group if level in available_levels_list and level not in used_levels]
        if unique:
            key = f"noise{'-'.join(str(l) for l in unique)}%"
            noise_dict[key] = unique
            used_levels.update(unique)

    return noise_dict

def get_electrodes_list(selected_subject, gyrus):
    """
    Get a list of electrode indices located in the specified gyrus for a selected subject.
    Parameters
    selected_subject : dict
        Dictionary containing the selected subject's data.
    gyrus : str
        Name of the gyrus to filter electrodes by.
    Returns
    -------
    electrodes_list : list
        List of electrode indices located in the specified gyrus.
    """
    assert gyrus in selected_subject['selected_subject_data']['gyrus'], f"Gyrus '{gyrus}' not found in subject's electrode data."
    selected_subject_data = selected_subject['selected_subject_data']
    electrodes_df = pd.DataFrame(data = selected_subject_data['gyrus'], columns = ['Gyrus'])
    electrodes_list = list(electrodes_df.query("Gyrus == @gyrus").index)
    return electrodes_list

def get_power_epochs(
    selected_subject,
    freq_down,
    freq_up,
    smooth_power,
    noise_levels_dict,
    epoch_start_wrt_image_onset = 0,
    epoch_end_wrt_image_onset = 1000,
    gyrus='Fusiform Gyrus',
    category='face',
    key_pressed=1,
    minimum_epochs_to_average=5,
    plot_averaged_power_by_noise=True,
    ylim=(-1, 8),
    figsize=(15, 5),
    zoom_electrode_idx=None,
):
    """
    Compute and optionally plot the averaged power epochs for a specified frequency band (in Hz)
    for electrodes in a specified gyrus, categorized by specified noise levels, for selected epochs 
    and for a selected subject.
    Parameters
    ----------
    selected_subject : dict
        Dictionary containing the selected subject's data.
    freq_down : int
        Lower bound of the frequency band for bandpass filtering (in Hz).
    freq_up : int
        Upper bound of the frequency band for bandpass filtering (in Hz).
    smooth_power : int or None
        Cutoff frequency for low-pass filtering the power signal (in Hz). If None, no smoothing is applied.
    noise_levels_dict : dict
        Dictionary where keys are string labels of noise level combinations and values are lists of the corresponding noise levels.
    epoch_start_wrt_image_onset : int, optional
        Start of the epoch relative to image onset (in ms). Default is 0.
    epoch_end_wrt_image_onset : int, optional
        End of the epoch relative to image onset (in ms). Default is 1000.
    gyrus : str, optional
        Name of the gyrus to filter electrodes by. Default is 'Fusiform Gyrus'.
    category : str, optional
        Category of images to consider ('face' or 'house'). Default is 'face'.
    key_pressed : int, optional
        Key press response to consider (1 for key pressed, 0 for no key press). Default is 1.
    minimum_epochs_to_average : int, optional
        Minimum number of epochs required to compute an average. Default is 5.
    plot_averaged_power_by_noise : bool, optional
        If True, plot the averaged power epochs by noise levels. Default is True.
    ylim : tuple, optional
        Y-axis limits for the plots. Default is (-1, 8).
    figsize : tuple, optional
        Size of the figure for the plots. Default is (15, 5).
    zoom_electrode_idx : tuple or None, optional
        If provided, should be a tuple (electrode_index, start_time, end_time) to zoom in on a specific electrode and time range in a separate plot. Default is None.
    Returns
    -------
    power_averaged_dict : dict
        Dictionary where keys are noise level labels and values are 2D arrays of averaged power epochs (time x electrodes).
    power_nonaveraged_dict : dict
        Dictionary where keys are noise level labels and values are 3D arrays of non-averaged power epochs (epochs x time x electrodes).
    """
    selected_subject_data = selected_subject['selected_subject_data']
    selected_subject_number = selected_subject['selected_subject_number']
    images_df = construct_images_df(selected_subject)
    power_all_electrodes = get_power_all_electrodes(selected_subject, freq_down, freq_up, smooth_power)
    n_channels = power_all_electrodes.shape[1]
    n_images = len(selected_subject_data['stim_id'])
    trange = np.arange(epoch_start_wrt_image_onset, epoch_end_wrt_image_onset)
    electrodes_list = get_electrodes_list(selected_subject, gyrus)

    # Get power of each epoch, for all electrodes :
    tepochs = selected_subject_data['t_on'][:, np.newaxis] + trange
    power_epochs = power_all_electrodes[tepochs, :]
    power_epochs = np.reshape(power_epochs, (n_images, trange.size, n_channels))

    # Get epochs for each noise category and average them :
    power_averaged_dict = {}
    power_nonaveraged_dict = {}
    nb_epochs = []
    for label, noise_vals in noise_levels_dict.items():
        mask = ((images_df['Category'] == category) &
                (images_df['Key_pressed'] == key_pressed) &
                (images_df['Noise'].isin(noise_vals)))
        epochs = power_epochs[mask]
        nb_epochs.append(epochs.shape[0])
        power_nonaveraged_dict[label] = epochs
        if epochs.shape[0] >= minimum_epochs_to_average :
            power_averaged_dict[label] = epochs.mean(axis=0)
        else:
            power_averaged_dict[label] = np.full((trange.size, n_channels), np.nan)

    # Plotting averaged power by noise :
    if plot_averaged_power_by_noise :
        colors = plt.get_cmap('tab10')(np.linspace(0.1, 0.9, len(power_averaged_dict)))
        n_cols = 4
        n_rows = int(np.ceil(len(electrodes_list) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
        axes = axes.flatten()

        for i, elec_idx in enumerate(electrodes_list):
            ax = axes[i]
            for k, (noise_lev, avr_power) in enumerate(power_averaged_dict.items()):
                ax.plot(trange, avr_power[:, elec_idx], color=colors[k], label=f"{noise_lev} ({nb_epochs[k]} epochs)")
            ax.set_title(f"Electrode {elec_idx+1}", fontsize=10)
            ax.set_ylim(ylim)
            ax.tick_params(labelsize=8)
            ax.set_xlabel('Time (ms)', fontsize=8)
            ax.set_ylabel('Amplitude', fontsize=8)    
            ax.tick_params(
                axis='both',        
                which='both',       
                labelbottom=True,
                labelleft=True,
                labeltop=False,
                labelright=False
            )

        for idx in range(len(electrodes_list), len(axes)):
            fig.delaxes(axes[idx])

        fig.suptitle(f"{freq_down}-{freq_up}Hz band power of averaged epochs of {gyrus} electrodes for correctly identified faces by subject {selected_subject_number}", fontsize=16)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.04), ncol=len(power_averaged_dict), fontsize=9)
        fig.tight_layout(rect=(0, 0.05, 1, 0.95))
        plt.show()

        # Zooming on one electrode :
        if zoom_electrode_idx is not None :
            elec_idx, t_start, t_end = zoom_electrode_idx
            num_elec = elec_idx + 1
            idx_in_layout = electrodes_list.index(elec_idx)
            ax_ref = axes[idx_in_layout]

            fig, ax = plt.subplots(figsize=(5.5, 3))
            for k, (noise_lev, power) in enumerate(power_averaged_dict.items()):
                ax.plot(trange[t_start:t_end], power[t_start:t_end, elec_idx], color=colors[k], label=f"{noise_lev} ({nb_epochs[k]} epochs)")
            
            ax.set_title(f"Zoom on electrode No. {num_elec}", fontsize=16)
            ax.set_xlabel("Time (ms)", fontsize=8)
            ax.set_ylabel("Amplitude", fontsize=8)
            ax.set_yticks(ax_ref.get_yticks())
            ax.set_ylim(ylim)
            ax.tick_params(labelsize=ax_ref.xaxis.label.get_size())
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(fontsize=7, loc='upper right')
            fig.tight_layout()
            plt.show()

    return power_averaged_dict, power_nonaveraged_dict
    

def get_power_epochs_oneelec(power_epochs_allelec, elec_idx) :
    """
    Extract power epochs per noise category for a single electrode from already computed power epochs of all electrodes.
    Parameters
    ----------
    power_epochs_allelec : dict
        Dictionary of power epochs (3D arrays of shape (n_epochs x n_timesteps x n_channels)) per noise category.
    elec_idx : int
        Index of the electrode of interest.
    Returns
    -------
    power_epochs_averaged_oneelec : dict
        Dictionary of averaged power epochs (1D arrays of shape (n_timesteps,)) per noise category for a single electrode.
    power_epochs_oneelec : dict
        Dictionary of power epochs (2D arrays of shape (n_epochs x timesteps)) per noise category for a single electrode.
    """
    power_epochs_oneelec = {}
    power_epochs_averaged_oneelec = {}
    for key, val in power_epochs_allelec.items():
        power_epochs_oneelec[key] = val[:,:,elec_idx] 
        power_epochs_averaged_oneelec[key] = val[:,:,elec_idx].mean(0) 
    return power_epochs_averaged_oneelec, power_epochs_oneelec


def get_FirstPeakPowerSlope_AveragedEpochs(power_epochs_averaged_oneelec, starting=150, height=2, prominence=0.05, plot=False) :
    """
    Calculate the slope of the ascending edge of the first peak in averaged power epochs for a single electrode across different noise categories.
    Parameters
    ----------
    power_epochs_averaged_oneelec : dict
        Dictionary of averaged power epochs (1D arrays of shape (n_timesteps,)) per noise category.
    starting : int, optional
        Starting index to consider for slope calculation (in ms). Default is 150.
    height : float, optional
        Minimum height of peaks to consider. Default is 2.
    prominence : float, optional
        Minimum prominence of peaks to consider. Default is 0.05.
    plot : bool, optional
        If True, plot the slopes for each noise category. Default is False.
    Returns
    -------
    slopes : dict
        Dictionary containing the slope of the ascneding edge of the first peak per noise category.
    """

    slopes = {}

    for key, val in power_epochs_averaged_oneelec.items() :
        avrg_epoch = val
        avrg_epoch = avrg_epoch[starting:]
        maxima_indices, _ = signal.find_peaks(avrg_epoch, height=height, prominence=prominence)
        minima_indices, _ = signal.find_peaks(-avrg_epoch, height=height, prominence=prominence)

        starting_idx = 0
        end_idx = len(avrg_epoch) - 1
        if len(maxima_indices) > 0:
            end_idx = maxima_indices[0]
        if len(minima_indices) > 0 and minima_indices[0] < end_idx:
            starting_idx = minima_indices[0]

        tgnt = np.diff(avrg_epoch)
        tgnt = tgnt[starting_idx:end_idx]

        slopes[key] = tgnt.mean().item()

    if plot:
        x_labels = list(slopes.keys())
        y_values = list(slopes.values())
        x_positions = range(len(x_labels))
        plt.figure(figsize=(5, 4))
        plt.scatter(x_positions, y_values, s=70)
        plt.xticks(x_positions, x_labels, rotation=20)
        plt.ylim(0, 0.05)
        plt.tight_layout()
        plt.show()
    return slopes

def get_FirstPeakPowerSlope_IndividualEpochs(power_epochs_oneelec, subject_idx, electrode_idx, starting=150, height=2, prominence=0.05, plot=True) :
    """
    Calculate the slope of the ascending edge of the first peak in individual power epochs for a single electrode across different noise categories.
    Parameters
    ----------
    power_epochs_oneelec : dict
        Dictionary of power epochs (2D arrays of shape (n_epochs x n_timesteps)) per noise category for a single electrode.
    subject_idx : int
        Index of the subject for labeling purposes.
    electrode_idx : int
        Index of the electrode for labeling purposes.
    starting : int, optional
        Starting index to consider for slope calculation (in ms). Default is 150.
    height : float, optional
        Minimum height of peaks to consider. Default is 2.
    prominence : float, optional
        Minimum prominence of peaks to consider. Default is 0.05.
    plot : bool, optional
        If True, plot the slopes for each noise category. Default is True.
    Returns
    -------
    slopes_dict : dict
        Dictionary of slopes of the ascending edge of the first peak per noise category.
    slopes_df : pandas.DataFrame
        DataFrame containing slopes and corresponding noise category labels.
    """
    
    slopes_dict = {}

    for key, val in power_epochs_oneelec.items() :
        slopes_onecat = []
        indiv_epochs = val
        for epoch in indiv_epochs:
            sig = epoch[starting:]
            maxima_indices, _ = signal.find_peaks(sig, height=height, prominence=prominence)
            minima_indices, _ = signal.find_peaks(-sig, prominence=prominence)

            starting_idx = 0
            end_idx = len(sig) - 1
            if len(maxima_indices) > 0:
                end_idx = maxima_indices[0]
            if len(minima_indices) > 0 and minima_indices[0] < end_idx:
                starting_idx = minima_indices[0]

            tgnt = np.diff(sig)
            tgnt = tgnt[starting_idx:end_idx]
            tgnt = tgnt.mean()

            slopes_onecat.append(tgnt)

        slopes_dict[key] = slopes_onecat

    all_slopes = []
    all_labels_str = []

    for cat, slopes in slopes_dict.items():
        clean_slopes = [s for s in slopes if not np.isnan(s)]
        all_slopes.extend(clean_slopes)
        all_labels_str.extend([cat] * len(clean_slopes))

    all_slopes = np.array(all_slopes)
    all_labels_str = np.array(all_labels_str)

    grouped_noise_labels = list(power_epochs_oneelec.keys())
    mapping = {cat : i for i, cat in enumerate(grouped_noise_labels)}
    all_labels_num = np.array([mapping[cat] for cat in all_labels_str])

    slopes_df = pd.DataFrame({
        "Slope": all_slopes,
        "Noise_Category_str": all_labels_str,
        "Noise_Category_num": all_labels_num,
    })

    if plot:
        df = pd.DataFrame({'Noise_Category': all_labels_str, 'Slope': all_slopes})
        plt.figure(figsize=(7, 5))
        sns.stripplot(data=df, x='Noise_Category', y='Slope', jitter=True, alpha=0.6, color='salmon', size=4)
        sns.boxplot(
            data=df,
            x="Noise_Category",
            y="Slope",
            whis=[2.5,97.5],
            width=0.4,
            showcaps=True,
            boxprops={'facecolor': 'none'},
            showfliers=False,
        )
        plt.xlabel("Noise Category", fontsize=12)
        plt.ylabel("Slope", fontsize=12)
        plt.title(f"First peak mean slope of the event-related high-gamma band power epochs (Subject {subject_idx+1}, Electrode {electrode_idx+1})", fontsize=14)
        plt.grid(True, axis='y')
        plt.xticks(rotation=0)
        plt.plot(0, 0, color='white', label="Whiskers = 2.5 - 97.5 percentiles")
        plt.legend(loc='upper right', fontsize='small')
        plt.tight_layout()
        plt.show()

    return slopes_dict, slopes_df


def resample_with_replacement(epochs_array):
    """
    Resample epochs with replacement.   
    Parameters
    ----------
    epochs_array : np.ndarray
        2D array of shape (epochs, time).
    Returns
    -------
    resampled_epochs : np.ndarray
        2D array of resampled epochs with the same shape as input.
    sample_idx : np.ndarray
        Array of rasampled indices.
    """
    sample_idx = np.random.choice(epochs_array.shape[0], size = epochs_array.shape[0], replace = True)
    resampled_epochs = epochs_array[sample_idx]
    return resampled_epochs, sample_idx

def get_bootstraps(power_epochs_oneelec, N=10000) :
    """
    Generate bootstrap samples for each noise category and compute slope of the ascending edge of the first peak of averaged power epochs.
    Parameters
    -------
    power_epochs_oneelec : dict
        Dictionary of power epochs (2D arrays of shape (n_epochs x n_timesteps)) per noise category for a single electrode.
    N : int, optional
        Number of bootstrap samples to generate. 10000 by default.
    Returns
    -------
    all_slopes : dict
        Dictionary of slopes of the ascending edge of the first peak per noise category, each containing a list of N slopes from averaged bootstrap epochs.
    """
    mean_power_bootstraps = {}
    all_slopes = {}
    for key, val in power_epochs_oneelec.items() :
      all_slopes[key] = []
      mean_power_all_bootstraps_onecat = np.zeros((N,1000))
      for i in range(N):
        resampled_epochs, _ = resample_with_replacement(val) 
        mean_power_one_bootstrap = resampled_epochs.mean(axis=0) #(1000,)
        mean_power_all_bootstraps_onecat[i] = mean_power_one_bootstrap  #(N, 1000)
      mean_power_bootstraps[key] = mean_power_all_bootstraps_onecat  
    
    for i in range(N):
      for key, val in mean_power_bootstraps.items() :
        one = {}
        one[key] = val[i]
        slopes = get_FirstPeakPowerSlope_AveragedEpochs(one)
        for key, val in slopes.items():
          all_slopes[key].append(val)

    return all_slopes

def plot_power_slope_averagedepochs_with_bootstrapping(power_slope_first_peak_averagedepochs, boostraps_slopes, subject_idx, electrode_idx):
    """
    Plot the mean slope of the ascending edge of the first peak in averaged power epochs with bootstrap confidence intervals for a single electrode across different noise categories.
    Parameters
    ----------
    power_slope_first_peak_averagedepochs : dict
        Dictionary of the slope of the ascending edge of the first peak per noise category for a single electrode, from averaged epochs.
    boostraps_slopes : dict
        Dictionary of slopes of the ascending edge of the first peak per noise category for a single electrode, from averaged bootstrap epochs.
    subject_idx : int
        Index of the subject for labeling purposes.
    electrode_idx : int
        Index of the electrode for labeling purposes.
    Returns
    -------
    None
    """
    slopes_all_extended = []
    noise_all_extended = []
    info = {}
    plt.figure(figsize=(6, 5))

    for idx, (key, val) in enumerate(power_slope_first_peak_averagedepochs.items()):
        info[key] = val
        plt.plot(idx, val, 'o', color='red', markersize=6, label="Measured slope" if idx == 0 else None)

    for key, val in boostraps_slopes.items():
        slopes_all_extended.extend(val)
        noise_all_extended.extend([key]*len(val))
        lower, upper = np.percentile(np.array(val), [2.5, 97.5])
        print(f"{key:20} : first peak slope = {info[key]:.4f} (95% CI {lower:.4f} - {upper:.4f})")

    df = pd.DataFrame({
        "Noise_Category": noise_all_extended,
        "Slope": slopes_all_extended
    })

    noise_cat = list(boostraps_slopes.keys())
    x_positions = range(len(noise_cat))

    sns.boxplot(
        data=df,
        x="Noise_Category",
        y="Slope",
        whis=[2.5,97.5],
        width=0.4,
        showcaps=True,
        boxprops={'facecolor': 'none'},
        medianprops={'color': 'black'},
        showfliers=False,
        order=noise_cat
    )

    plt.xlabel("Noise Category", fontsize=12)
    plt.ylabel("Slope", fontsize=12)
    plt.title(f"First peak mean slope of the averaged event-related high-gamma band power (Subject {subject_idx+1}, Electrode {electrode_idx+1})", fontsize=14)
    plt.plot(0, 0, color='white', label="\nWhiskers = 2.5 - 97.5 percentiles")
    plt.xticks(x_positions, noise_cat, rotation=20)
    plt.grid(True, axis='y')
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()

def get_FirstPeakPowerLatency_IndividualEpochs(power_epochs_oneelec, subject_idx, electrode_idx, seed_examples, starting=150, height=2, prominence=0.05, plot_examples=False, plot=True) :
    """
    Calculate the latency of the first peak in individual power epochs for a single electrode across different noise categories.
    "Latency" is defined as the number of timesteps between image onset and the first peak of individual epochs.
    Parameters
    ----------
    power_epochs_oneelec : dict
        Dictionary of power epochs (2D arrays of shape (n_epochs x n_timesteps)) per noise category for a single electrode.
    subject_idx : int
        Index of the subject for labeling purposes.
    electrode_idx : int
        Index of the electrode for labeling purposes.
    seed_examples : int
        Random seed for selecting example epochs to plot.
    starting : int, optional
        Starting index to consider for latency calculation (in ms). Default is 150.
    height : float, optional
        Minimum height of peaks to consider. Default is 2.
    prominence : float, optional
        Minimum prominence of peaks to consider. Default is 0.05.
    plot_examples : bool, optional
        If True, plot example epochs with detected peaks. Default is False.
    plot : bool, optional
        If True, plot the latencies for each noise category. Default is True.
    Returns
    -------
    latencies_dict : dict
        Dictionary of latencies of the first peak per noise category, each containing a list of latencies from individual epochs with their mean and standard deviation.
    latencies_df : pandas.DataFrame
        DataFrame containing latencies and corresponding noise category labels.
    """
    latencies_dict = {}

    for key, val in power_epochs_oneelec.items() :
        latencies_onecat = []
        indiv_epochs = val
        for epoch in indiv_epochs:
            sig = epoch[starting:]
            peak_indices, _ = signal.find_peaks(sig, height=height, prominence=prominence)

            if len(peak_indices) == 0:
                latencies_onecat.append(np.nan)
            if len(peak_indices) > 0:
                first_peak_idx = peak_indices[0] + starting
                latencies_onecat.append(first_peak_idx)

        latencies_onecat = np.array(latencies_onecat)
        mean_latencies_onecat = np.nanmean(latencies_onecat)
        std_latencies_onecat = np.nanstd(latencies_onecat)

        latencies_dict[key] = latencies_onecat, mean_latencies_onecat, std_latencies_onecat

    all_lat = []
    all_labels_str = []

    for cat, val in latencies_dict.items():
        lat = val[0]
        clean_lat = [l for l in lat if not np.isnan(l)]
        all_lat.extend(clean_lat)
        all_labels_str.extend([cat] * len(clean_lat))

    all_lat = np.array(all_lat)
    all_labels_str = np.array(all_labels_str)

    grouped_noise_labels = list(power_epochs_oneelec.keys())
    mapping = {cat : i for i, cat in enumerate(grouped_noise_labels)}
    all_labels_num = np.array([mapping[cat] for cat in all_labels_str])

    latencies_df = pd.DataFrame({
        "Latency": all_lat,
        "Noise_Category_str": all_labels_str,
        "Noise_Category_num": all_labels_num,
    })


    if plot_examples :
        trange = np.arange(1000)
        noise_cat = list(power_epochs_oneelec.keys())
        n_col = len(power_epochs_oneelec)
        n_row = 3 if min(len(v) for v in power_epochs_oneelec.values())>= 3 else min(len(v) for v in power_epochs_oneelec.values())

        fig, axes = plt.subplots(n_row, n_col, figsize=(3.5*n_col, 2.8*n_row), squeeze=False)
        colors = plt.get_cmap('tab10')(np.linspace(0.1, 0.9, n_col))
    
        rng = np.random.default_rng(seed_examples)

        for col, cat in enumerate(noise_cat):
            epochs = power_epochs_oneelec[cat]
            color = colors[col]
            indices = rng.choice(len(epochs), size=n_row, replace=False)
            rows = epochs[indices]

            for i, row in enumerate(rows):
                ax = axes[i, col]
                peak = latencies_dict[cat][0][indices[i]]
                if peak != np.nan: 
                    ax.axvline(peak, color=color, linestyle='--', alpha=0.7)
                ax.plot(trange, row, color=color, linewidth=1.2)
                ax.set_ylim(-2,10)
                
                if i == 0:
                    ax.set_title(cat, fontsize=12)
                if col==0 :
                    ax.set_ylabel("Power", fontsize=12)
                if i == n_row - 1:
                    ax.set_xlabel("Time (ms)", fontsize=12)
                if i != n_row - 1:
                    ax.set_xticklabels([])
                if col != 0:
                    ax.set_yticklabels([])
                ax.grid(True)

        fig.suptitle(f"Examples of Event-Related high-gamma band power epochs (first peak indicated) (Subject {subject_idx+1}, Electrode {electrode_idx+1})", fontsize=14)
        plt.show()

    if plot:
        df = pd.DataFrame({'Noise_Category': all_labels_str, 'Latency': all_lat})
        plt.figure(figsize=(7, 5))
        sns.stripplot(data=df, x='Noise_Category', y='Latency', jitter=True, alpha=0.6, color='salmon', size=4)
        sns.boxplot(
            data=df,
            x="Noise_Category",
            y="Latency",
            whis=[2.5,97.5],
            width=0.4,
            showcaps=True,
            boxprops={'facecolor': 'none'},
            showfliers=False,
        )
        plt.xlabel("Noise Category")
        plt.ylabel("Latency (ms)")
        plt.title(f"First peak latency of the event-related high-gamma band power epochs (Subject {subject_idx+1}, Electrode {electrode_idx+1})", fontsize=14)
        plt.grid(True, axis='y')
        plt.plot(0, np.mean(all_lat), color='white', label="Whiskers = 2.5 - 97.5 percentiles")
        plt.legend(loc='upper right', fontsize='small')
        plt.tight_layout()
        plt.show()

    return latencies_dict, latencies_df


def spearman_permutation_test(power_feature_individualepochs_df, feature_str, n_perm, seed):
    """
    Perform a permutation test to assess the significance of the Spearman correlation
    between a specified feature of individuals power epochs for one single electrode 
    and noise categories.
    Parameters
    ----------
    power_feature_individualepochs_df : pandas.DataFrame
        DataFrame containing the feature values and corresponding noise category labels.
    feature_str : str
        Name of the feature to test. 
        Can be either 'Slope' : the mean slope of the ascending edge of the first peak of individual epochs.
        Or 'Latency' : the number of timesteps between image onset and the first peak of individual epochs.
    n_perm : int
        Number of permutations to perform.
    seed : int
        Random seed for reproducibility.
    Returns
    -------
    corr_observed : float
        Observed Spearman correlation between the specified feature of individuals power epochs and noise categories.
    p_value : float
        p-value from the permutation test.
    """
    df = power_feature_individualepochs_df
    rng = np.random.default_rng(seed)
    perm_corrs = []

    corr_observed, _ = spearmanr(df[feature_str], df['Noise_Category_num'])
    
    for _ in range(n_perm):
        shuffled_labels = rng.permutation(df['Noise_Category_num'])
        corr_perm, _ = spearmanr(df[feature_str], shuffled_labels)
        perm_corrs.append(corr_perm)

    perm_corrs = np.array(perm_corrs)
    p_value = np.mean(np.abs(perm_corrs) >= np.abs(corr_observed))

    print(f"Observed Correlation : {corr_observed:.4f}")
    print(f"p-value permutation ({n_perm} permutations) : {p_value:.5f}")

    return corr_observed, p_value
