
### Notes on Improvements Needed

* Break Up Utils something like (makes things cleaner):
    - Surface Utils
    - Filament Utils
    - Spot Utils
    - I/O Utils

* Create the testing suite.
    - This makes making code changes easy because we can quickly run a test to confirm it works. 


# keywords

surface
surface objects
surface tracks
surface track objects

filaments
filament objects

spots
spot tracks
spot track objects

# notes
when we are looking for objects some objects can be there that belong to a track
or sometimes there can be objects but they wont be connected with a track. 
our system needs to handle this. an example is in spot_track_object_parser 

# questions
when we extract surface object information that is just objects that are created that does not belong to a track right?
if we do this when we extract surface track objects what we are doing is from the same set of surface objects we are grabbing
    only the ones that belong to track, but like in the spot track object parser if there is no tracks we are still grabbing the objects that dont belong to a track. is this right? should we be grabbing these items because they dont have track statistics.

*** look at the the stats names for objects vs track_objects .. and see what the difference is **

*** is there a difference between objects that belong to a track vs objects that dont belong to a track 
in terms of how they are represented in the ims file. for example track_objects are in a seperate place 
if there something that indicates in the statistics value dataframe that says these objects belong to a track
and these objects dont? **

** for the time step parsers .. do objects that dont belong to a track have first time points? **

# TODO:
    * Surface time step parser needs the script and need to update the surface_parser notebook to run that script. 
    * Also the time step parser is based on tracks right? without tracks there cannot be time steps right?