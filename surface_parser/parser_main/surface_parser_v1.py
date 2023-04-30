import utils
import copy


def extract_and_save(ims_file_path: str, 
                     valid_surface: int, 
                     categories_list: list, 
                     save_path: str) -> None:
    
    # load the imaris file
    data = utils.load_ims(ims_file_path)
    
    try:
        
        # get surface we want to parse
        surface_name = utils.get_object_names(full_data_file=data, 
                                            search_for='Surface')[valid_surface]
        
        # get all the statistics names
        surface_stats_names = utils.get_statistics_names(full_data_file=data, object_name=surface_name)
        
        # get the statistics values in the surface
        surface_stats_values = utils.get_stats_values(full_data_file=data, object_name=surface_name)
        
        # create a empty dict where key=numeric stats ids and value = None
        # this dictionary is simply a container to store all the statistics values
        empty_stats_dict = {key: None for key in surface_stats_names.keys()}   
        
        # create a empty dict where key=object_id, and value=empty stats dict
        # this dictionary is a container where for each object in the surface ..
        # .. it contains another dictionary that stores all the stats values
        empty_data_dict = {key: copy.deepcopy(empty_stats_dict) for key in surface_stats_values['ID_Object']}
        
        # start data extraction in the loop below
        for index in range(len(surface_stats_values)):
            
            # get the current data points 
            current_data = surface_stats_values.iloc[index]
            
            # get the object id the data is associated with
            object_id = current_data['ID_Object']
            
            # get the type of the value
            stats_type = current_data['ID_StatisticsType']
            
            # get the statistics value
            value = current_data['Value']
            
            # insert current selection into dictionary
            try:
                empty_data_dict[object_id][stats_type] = value
            except KeyError:
                # key error occurs if for a stats name there is no value
                # missing values will be represented as None
                pass
            
        # invert dictionary + name modifications
        # this step is a cosmetic step
        inverted_stats_names = utils.invert_stats_dict(surface_stats_names)
        inverted_stats_names = utils.flatten(inverted_stats_names)
        
        # create a list of stats names (in integer form) we want to remove
        del_list = utils.create_del_list(inverted_stats_names, categories_list)
        
        # reverse the stats names again such that key=num, value=name
        final_stats_names = {v: k for k,v in inverted_stats_names.items()}
        
        # generate csv
        dataframe = utils.generate_csv(data_dict=empty_data_dict, 
                                    del_list=del_list,
                                    stats_names=final_stats_names,
                                    categories_list=categories_list)
        
        
        # save the dataframe in the same location
        dataframe.to_csv(save_path)
        
    except AttributeError:
        print(f"skipping file {ims_file_path} -- no surface found")
