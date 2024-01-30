import os

def my_paths(server, db):
    
    if server == 'local':
        output_dir = '/mnt/PS6T/OUTPUTS'
        data_dir = fetch_linux_db(db)
    else:
        raise NotImplementedError(f"{server} is not configured")
    
    return output_dir, data_dir


def fetch_linux_db(db):
    
    root="/mnt/PS6T/datasets/"
    
    paths = {
        'kinetics700': root+'Video/kinetics/kinetics700',
        'kinetics400': root+'Video/kinetics/kinetics400',
        }
    
    try:
        return paths[db]   
    except:
        raise NotImplementedError(f'{db} is not available in local.')
        

  
        
