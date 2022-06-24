
.. |br| raw:: html

   <br /><br />


***********
Data Models
***********

All functions in the processing function library should use these data models for the input and output variables. When referring to the dimensions (X, Y, Z) the Z dimension is the fastest varying while X is the slowest varying dimension. 
 
Visibility functions
====================
Functions which would be working with visibilities should use these input/outputs. Only parameters which are used be the function should be passed, not all of them.

- *Visibilities:* ``sdp_Mem *vis`` 

  - **Type:** complex-valued array

  - **Dimensions:** (time samples, baselines, channels, polarizations)
    
  
- *UVW coordinates:* ``sdp_Mem *uvw`` [metres]

  - **Type:** real-valued array

  - **Dimensions:** (time samples, baselines, 3)
    
  
- *Weights:* ``sdp_Mem *weights`` 

  - **Type:** real-valued array
  
  - **Dimensions:** (time samples, baselines, channels)
    
  
- *Channel start:* ``double channel_start`` [Hz]

  - **Type:** real-valued variable
    
  
- *Channel step:* ``double channel_step`` [Hz]

  - **Type:** real-valued array
    
  
- *Time centroids:* ``sdp_Mem *time_centroids`` [s]
 
  - **Type:** real-valued array
  
  - **Dimensions:** (time samples, baselines, channels)
    
  
- *Freq centroids:* ``sdp_Mem *freq_centroid`` [Hz]
 
  - **Type:** real-valued array
  
  - **Dimensions:** (time samples, baselines, channels)
  
  
- *Exposure:* ``sdp_Mem *exposure`` [s]
 
  - **Type:** real-valued array
  
  - **Dimensions:** (time samples, baselines, channels)
    
  
- *Bandwidth:* ``sdp_Mem *bandwidth`` [Hz]
 
  - **Type:** real-valued array
  
  - **Dimensions:** (time samples, baselines, channels)
  
  
- *Phase centre Right Ascension:* ``double phasecentre_ra_rad`` [radians]

  - **Type:** real-valued variable
    
  
- *Phase centre Declination:* ``double phasecentre_dec_rad`` [radians] 

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



Grid functions
==============
Functions which would be working on grids should use these input/outputs. Only parameters which are used be the function should be passed, not all of them.


- *Grid:* ``sdp_Mem *grid`` // double complex [chan][w][v][u][polar]

  - **Type:** real-valued array
  
  - **Dimensions:** (channels, w, v, u, polarizations)


- *Coordinate step:* ``double du, double dv, double dw``

  - **Type:** real-valued
  
  - **Note:** Do we want a sdp_Mem array for that? It would deal with the precision.
  
  
- *Grid offset:* ``double offset_u, double offset_v, double offset_w``

  - **Type:** real-valued
  
  - **Note:** Do we want a sdp_Mem array for that? It would deal with the precision.


- *Phase centre Right Ascension and Declination:* ``double phasecentre_ra_rad, double phasecentre_dec_rad`` [radians]

  - **Type:** real-valued variable
    
  
- *UVW Projection:* ``sdp_Mem *uvw_projection``

  - **Type:** real-valued
  
  - **Dimensions:** (3, 3)
  
  
- *Polarisation metadata:* ``sdp_Table *polar``   

  - **Note:** table-like data type not implemented yet
    
  
  
UVW functions
=============
Functions which would be working on UVW coordinates should use these input/outputs. Only parameters which are used be the function should be passed, not all of them.

- *UVW coordinates:* ``sdp_Mem *uvw`` 

  - **Type:** real-valued array

  - **Dimensions:** (time samples, baselines, 3)
    
  
- *Time:* ``double time_start, double time_step``

  - **Type:** real-valued
  
  
- *Phase centre Right Ascension and Declination:* ``double phasecentre_ra_rad, double phasecentre_dec_rad`` [radians]

  - **Type:** real-valued variable
  
  
- *UVW Projection:* ``sdp_Mem *uvw_projection``

  - **Type:** real-valued
  
  - **Dimensions:** (3, 3)
  
  
- *Baseline metadata:* ``sdp_Table *bl``

  - **Note:** table-like data type not implemented yet


Image functions
===============
Functions which would be working on UVW coordinates should use these input/outputs. Only parameters which are used be the function should be passed, not all of them.

- *Image:* ``sdp_Mem *image`` 

  - **Type:** complex-valued array

  - **Dimensions:** (channels, m, l, polarizations)
  
  
- *Coordinate step:* ``double dl, double dm``

  - **Type:** real-valued
  

- *Phase centre Right Ascension and Declination:* ``double phasecentre_ra_rad, double phasecentre_dec_rad`` [radians]

  - **Type:** real-valued variable
  
  
- *lmn projection:* ``sdp_Mem *lmn_projection``

  - **Type:** real-valued
  
  - **Dimensions:** (3, 3)


- *Polarisation metadata:* ``sdp_Table *polar``   

  - **Note:** table-like data type not implemented yet
    
  
