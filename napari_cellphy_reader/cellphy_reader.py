"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the ``napari_get_reader`` hook specification, (to create
a reader plugin) but your plugin may choose to implement any of the hook
specifications offered by napari.
see: https://napari.org/docs/plugins/hook_specifications.html

Replace code below accordingly.  For complete documentation see:
https://napari.org/docs/plugins/for_plugin_developers.html
"""
import numpy as np
import dask.array as da
import dask
from napari_plugin_engine import napari_hook_implementation
from aicsimageio import AICSImage, imread

color_maps = ["bop purple", "bop orange", "bop blue", "green", "blue"]

SUPPORTED_FORMATS = ('.lif', '.czi')

@napari_hook_implementation
def napari_get_reader(path):
    """A basic implementation of the napari_get_reader hook specification.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    # we are not loading multiple files at a time
    
    if isinstance(path, str) and path.endswith(SUPPORTED_FORMATS):
        return reader_function

    return None


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of layer.
        Both "meta", and "layer_type" are optional. napari will default to
        layer_type=="image" if not provided
    """
    print("reading file ", path)
    aics_img = AICSImage(path)

    # dims are normaly in "STCZYX"
    number_of_channels = aics_img.size_c
    number_of_time_points = aics_img.size_t
    nz = aics_img.size_z
    ny = aics_img.size_y
    nx = aics_img.size_x
    name_of_channels = aics_img.get_channel_names()
    pixel_x, pixel_y, pixel_z = aics_img.get_physical_pixel_size()
    scale = [1, pixel_z, pixel_y, pixel_x]

    print("number_of_channels", number_of_channels)
    print("number_of_time_points", number_of_time_points)
    print("name_of_channels", name_of_channels)
    print("scale", scale)
    print("nz", nz)
    layer_list = []
    channel_dict = {}
    # for channel in name_of_channels:
    #     channel_dict[channel] = {}

    if number_of_channels > 1:
        print("number_of_channels > 1")
        for cindex , channel_name in enumerate(name_of_channels):
            if number_of_time_points > 1:
                print("number_of_time_points > 1")
                if nz > 1:
                    arr = da.stack(
                        [
                            aics_img.get_image_dask_data('ZYX', S=0, C=cindex, T=tindex)
                            for tindex in range(number_of_time_points)
                        ]
                    )
                else:
                    arr = da.stack(
                        [
                            aics_img.get_image_dask_data('YX', S=0, C=cindex, T=tindex, Z=0)
                            for tindex in range(number_of_time_points)
                        ]
                    )
                    scale = [1, pixel_y, pixel_x]
            else :
                
                if nz > 1:
                    arr = aics_img.get_image_dask_data('ZYX', S=0, C=cindex, T=0)
                    scale = [pixel_z, pixel_y, pixel_x]
                else:
                    print("number_of_time_points < 1")
                    print("nz < 1")
                    print("cindex: ", cindex)
                    print("channel_name: ", channel_name)
                    arr = aics_img.get_image_dask_data('YX', S=0, C=cindex, T=0, Z=0)
                        
                    scale = [pixel_y, pixel_x]
                    print("arr.shape",arr.shape)

            

            channel_dict[channel_name] = {
                "data" : dask.optimize(arr)[0],
                "colormap": color_maps[cindex % len(color_maps)]
            }
            
    else:
        if number_of_time_points > 1:
            if nz > 1:
                arr = da.stack(
                    [
                        aics_img.get_image_dask_data('ZYX', S=0, C=0, T=tindex)
                        for tindex in range(number_of_time_points)
                    ]
                )
            else:
                arr = da.stack(
                    [
                        aics_img.get_image_dask_data('YX', S=0, C=0, T=tindex, Z=0)
                        for tindex in range(number_of_time_points)
                    ]
                )
                scale = [1, pixel_y, pixel_x]
        else :
            if nz > 1:
                arr = aics_img.get_image_dask_data('ZYX', S=0, C=0, T=0)
                scale = [pixel_z, pixel_y, pixel_x]
            else:
                arr = aics_img.get_image_dask_data('YX', S=0, C=0, T=0, Z=0)
                scale = [pixel_y, pixel_x]

            

            channel_dict[channel_name] = {
                "data" : dask.optimize(arr)[0],
                "colormap": color_maps[0]
            }


    for channel_name, channel in channel_dict.items():
        print("creating layer channel_name", channel_name)
        add_kwargs = {
            "name": channel_name,
            "blending" : 'additive',
            "rendering" : "mip",
            "scale": scale,
            "colormap": channel['colormap']
        }
        print("channel['data'].shape", channel['data'].shape)
        layer_list.append(
            (
            channel['data'], #data
            add_kwargs, # kwargs
            "image" # layer type
        )
        )

    
            

    return layer_list

