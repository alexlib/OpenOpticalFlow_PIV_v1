import os
import time
import pickle
from typing import List, Tuple, Dict, Any

def piv_manage_jobs(im1: List[str], im2: List[str], piv_par_in: Dict[str, Any]) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Distribute the treatment of image sequence into several jobs and outputs settings for the first untreated job.

    Args:
        im1 (List[str]): List of paths to the first image in each image pair.
        im2 (List[str]): List of paths to the second image in each image pair.
        piv_par_in (Dict[str, Any]): Parameters defining the evaluation.

    Returns:
        Tuple[List[str], List[str], Dict[str, Any]]: Lists of images to be treated and parameters for the next job.
    """
    if not piv_par_in.get('anOnDrive', False):
        raise ValueError("Job management can be used only with option anOnDrive = true.")

    target_path = piv_par_in['anTargetPath']
    if not os.path.exists(target_path):
        try:
            os.makedirs(target_path)
        except Exception as e:
            raise FileNotFoundError(f"Target folder does not exist. Failed to create it: {e}")

    lock_master_wait = os.path.join(target_path, 'lock_Master_Wait.lck')
    lock_expiration_time = piv_par_in['jmLockExpirationTime']

    # Check the presence of lock_Master_Wait file
    while file_age(lock_master_wait) < lock_expiration_time:
        print("Job distribution file is locked. Waiting for unlocking...")
        time.sleep(1)

    # Create the lock file with _Wait status
    with open(lock_master_wait, 'w') as f:
        f.write('Distributing/identifying tasks.')

    # Read JobList, if the file exists, or initialize this variable
    job_list_path = os.path.join(target_path, 'JobList.mat')
    if os.path.exists(job_list_path):
        with open(job_list_path, 'rb') as f:
            job_list = pickle.load(f)
        os.remove(job_list_path)
        if len(job_list['ShouldStart']) != piv_par_in['jmParallelJobs']:
            job_list = {}
    else:
        job_list = {
            'lockFiles': [None] * piv_par_in['jmParallelJobs'],
            'ShouldStart': [True] * piv_par_in['jmParallelJobs'],
            'im1': [None] * piv_par_in['jmParallelJobs'],
            'im2': [None] * piv_par_in['jmParallelJobs']
        }

    # Test, if jobs should be redistributed
    redistribute_jobs = any(not job_list['ShouldStart'][i] and
                             (not os.path.exists(job_list['lockFiles'][i]) or
                              file_age(job_list['lockFiles'][i]) > lock_expiration_time)
                             for i in range(len(job_list['ShouldStart'])))

    if redistribute_jobs:
        # Distribute jobs (if necessary)
        output_list = [f for f in os.listdir(target_path) if f.startswith('piv') and f.endswith('.mat')]
        required_list = [f'piv_{treat_img_path(im1[i])[1]}_{treat_img_path(im2[i])[1]}.mat' for i in range(len(im1))]
        missing_i = [i for i, file in enumerate(required_list) if file not in output_list]

        missing_im1 = [im1[i] for i in missing_i]
        missing_im2 = [im2[i] for i in missing_i]
        missing_pairs = len(missing_i)

        if missing_pairs < piv_par_in['jmParallelJobs']:
            job_list['ShouldStart'][missing_pairs+1:] = [False] * (piv_par_in['jmParallelJobs'] - missing_pairs)

        # Distribute image pairs to jobs
        for kk in range(piv_par_in['jmParallelJobs'], 0, -1):
            if not job_list['ShouldStart'][kk]:
                continue
            start_i = (kk - 1) * (missing_pairs - 1) // piv_par_in['jmParallelJobs'] + 1
            stop_i = kk * (missing_pairs - 1) // piv_par_in['jmParallelJobs'] + 1
            job_list['im1'][kk] = missing_im1[start_i:stop_i]
            job_list['im2'][kk] = missing_im2[start_i:stop_i]
            job_list['lockFiles'][kk] = os.path.join(target_path, f'lock_Job_{kk:03d}.lck')
            if os.path.exists(job_list['lockFiles'][kk]):
                os.remove(job_list['lockFiles'][kk])

        # If Job001 is to be started and ~anPairsOnly, attribute to it all image pairs
        if job_list['ShouldStart'][1] and not piv_par_in.get('anPairsOnly', False):
            job_list['im1'][1] = im1
            job_list['im2'][1] = im2
            job_list['lockFiles'][1] = os.path.join(target_path, 'lock_Job_001.lck')
            if os.path.exists(job_list['lockFiles'][1]):
                os.remove(job_list['lockFiles'][1])

    # Start last unstarted task from the list
    start_indices = [i for i, start in enumerate(job_list['ShouldStart']) if start]
    if start_indices:
        job_index = start_indices[-1]
        piv_par_out = piv_par_in.copy()
        piv_par_out['seqJobNumber'] = job_index
        piv_par_out['jmLockFile'] = job_list['lockFiles'][job_index]
        if job_index > 1:
            piv_par_out['anPairsOnly'] = True
        im1out = job_list['im1'][job_index]
        im2out = job_list['im2'][job_index]
        with open(job_list['lockFiles'][job_index], 'w') as f:
            f.write('Starting task.')
        job_list['ShouldStart'][job_index] = False
        with open(job_list_path, 'wb') as f:
            pickle.dump(job_list, f)
    elif file_age(os.path.join(target_path, 'lock_Job_001.lck')) > lock_expiration_time:
        piv_par_out = piv_par_in.copy()
        piv_par_out['seqJobNumber'] = 1
        piv_par_out['jmLockFile'] = os.path.join(target_path, 'lock_Job_001.lck')
        im1out = im1
        im2out = im2
        with open(piv_par_out['jmLockFile'], 'w') as f:
            f.write('Starting task.')
    else:
        piv_par_out = {}
        im1out = []
        im2out = []
        print('Treatment of all image pairs is attributed to jobs. No treatment will occur.')
        print('    (User might want to erase lock files in the output folder.)')

    # Remove the lock file with _Wait status
    time.sleep(0.5)
    if os.path.exists(lock_master_wait):
        os.remove(lock_master_wait)

    return im1out, im2out, piv_par_out

def treat_img_path(path: str) -> Tuple[int, str, str]:
    """
    Separate the path to get the folder, filename, and number if contained in the name.

    Args:
        path (str): The file path.

    Returns:
        Tuple[int, str, str]: Image number, filename, and folder.
    """
    folder, filename = os.path.split(path)
    filename, ext = os.path.splitext(filename)
    img_no = ''.join(filter(str.isdigit, filename))
    return int(img_no) if img_no else None, filename, folder

def file_age(filename: str) -> float:
    """
    Calculate the age of a file in seconds.

    Args:
        filename (str): The file path.

    Returns:
        float: Age of the file in seconds.
    """
    if not os.path.exists(filename):
        return float('inf')
    return time.time() - os.path.getmtime(filename)
