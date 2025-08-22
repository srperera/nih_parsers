
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