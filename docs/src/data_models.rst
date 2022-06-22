
.. |br| raw:: html

   <br /><br />


***********
Data Models
***********

All functions in the processing function library should use these data models for the input and output variables. When referring to the dimensions [X][Y][Z] the Z dimension is the fastest varying while X is the slowest varying dimension. 
 
Visibility functions
====================
Functions which would be working with visibilities should use these input/outputs. Only parameters which are used be the function should be passed, not all of them.

- *Visibilities:* ``sdp_Mem *vis`` 

  - **Type:** complex-valued array

  - **Dimensions:** (time samples, baselines, channels, polarizations)
    
  
- *UVW coordinates:* ``sdp_Mem *uvw`` 

  - **Type:** real-valued array

  - **Dimensions:** (time samples, baselines, channels, 3)
    
  
- *Weights:* ``sdp_Mem *weights`` 

  - **Type:** real-valued array
  
  - **Dimensions:** (time samples, baselines, channels)
    
  
- *Channel start:* ``double channel_start`` 

  - **Type:** real-valued variable
    
  
- *Channel step:* ``double channel_step`` 

  - **Type:** real-valued array
    
  
- *Time centroids:* ``sdp_Mem *time_centroids``
 
  - **Type:** real-valued array
  
  - **Dimensions:** (time samples, baselines, channels)
    
  
- *Freq centroids:* ``sdp_Mem *freq_centroid``
 
  - **Type:** real-valued array
  
  - **Dimensions:** (time samples, baselines, channels)
  
  
- *Exposure:* ``sdp_Mem *exposure``
 
  - **Type:** real-valued array
  
  - **Dimensions:** (time samples, baselines, channels)
    
  
- *Bandwidth:* ``sdp_Mem *bandwidth``
 
  - **Type:** real-valued array
  
  - **Dimensions:** (time samples, baselines, channels)
    
- *Phase centre (radians):* ``double phasecentre_ra`` 

  - **Type:** real-valued variable
    
  
- *Phase centre (degrees):* ``double phasecentre_dec`` 

  - **Type:** real-valued array
    
  
- *Baseline metadata:* ``sdp_Table *bl``

  - **Note:** table-like data type not implemented yet
   
- *Polarisation metadata:* ``sdp_Table *polar``   

  - **Note:** table-like data type not implemented yet
  
- *Scan id:* ``sdp_Mem *scan_id``
  
  - **Type:** integer-valued array
  
  - **Dimensions:** (time samples)

  
- *Scan metadata:* ``sdp_Table *scan``

  - **Note:** table-like data type not implemented yet
























