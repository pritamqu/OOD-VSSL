import os

def return_home(server):

    if server == 'local':
        return '/home/pritam/workdir'
    else:
        raise ValueError

def my_paths(server, db):
    
    if server == 'local':
        output_dir = '/mnt/PS6T/OUTPUTS'
        data_dir = fetch_linux_db(db)
    
    return output_dir, data_dir
        
def fetch_linux_db(db):
    
    root="/mnt/PS6T/datasets/"
    
    paths = {
        'audioset': root+'AudioSet',
        'kinetics700': root+'Video/kinetics/kinetics700',
        'kinetics400': root+'Video/kinetics/kinetics400',
        'kinetics_sound': root+'Video/kinetics/kinetics400',
        'mimetics': root+'Video/mimetics',    
        'tinyvirat': root+'Video/TinyVirat/TinyVIRAT_V2',    
        'rareact': root+'Video/RareAct',    
        'sims4action': root+'Video/Sims4Action',    
        'actor_shift': root+'Video/ActorShift',    
        'nec_drone': root+'Video/nec_drone',    
        'charadesego': root+'Video/charades_ego',    
        'mitv2': root+'Video/moments_in_time/Moments_in_Time_Raw',    
        'ucf101': root+'Video/ucf101',
        'ucf-hmdb': root+'Video/ucf101',
        'afd101': root+'Video/afd101',
        'ucf101_flow': root+'Video/ucf101_flow',
        'hmdb51': root+'Video/hmdb51',
        'hmdb-ucf': root+'Video/hmdb51',
        'esc50': root+'Audio/ESC-50',
        'dcase': root+'Audio/DCASE',
        'ssv2': root+'Video/ssv2',
        'charades': root+'Video/charades_video',
        'epic_kitchen': root+'Video/epic-kitchen',
        'fsd50': root+'Audio/FSD50K',
        'openmic': root+'Audio/openmic-2018',
        'toyboxvideo': root+'Video/toybox_video',
        'coil100': root+'image/coil-100',
        'coil20': root+'image/coil-20',
        'epfl_car': root+'image/epfl_gims08',
        'stl10': root+'image',
        'nc20': root+'image/NCDataset/Gray', 
        'aloi': root+'image/aloi-viewpoint', 
        }
    
    try:
        return paths[db]   
    except:
        raise NotImplementedError(f'{db} is not available in local.')
        
