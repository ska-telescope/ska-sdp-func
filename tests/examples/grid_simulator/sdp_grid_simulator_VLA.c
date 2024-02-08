/*
 * sdp_grid_simulator_VLA.c
 *
 *  Created on: 5 Feb 2024
 *      Author: Vlad Stolyarov
 */

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cufftXt.h>

#include "sdp_grid_simulator_VLA.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"


void xyz_to_uvw(
        const double xyz[][3],
        double ha,
        double dec,
        double* uvw_ha,
        int nants
)
{
    int i;
    double x, y, z, v0;
    double u, v, w;
    for (i = 0; i < nants; ++i)
    {
        x = xyz[i][0];
        y = xyz[i][1];
        z = xyz[i][2];
        u = x * cos(ha) - y * sin(ha);
        v0 = x * sin(ha) + y * cos(ha);
        w = z * sin(dec) - v0 * cos(dec);
        v = z * cos(dec) + v0 * sin(dec);
        uvw_ha[3 * i] = u;
        uvw_ha[3 * i + 1] = v;
        uvw_ha[3 * i + 2] = w;
    }
}


void baselines(double* ants_uvw, double* basel_uvw, int nants )
{
    int i, j, k;
    // int basel_num;
    // basel_num = (int)(nants*(nants-1)/2);
    // printf("Baseline number is %d\n", basel_num);
    k = 0;
    for (i = 0; i < nants; ++i)
        for (j = i + 1; j < nants; ++j)
        {
            basel_uvw[3 * k] = ants_uvw[3 * j] - ants_uvw[3 * i];
            basel_uvw[3 * k + 1] = ants_uvw[3 * j + 1] - ants_uvw[3 * i + 1];
            basel_uvw[3 * k + 2] = ants_uvw[3 * j + 2] - ants_uvw[3 * i + 2];
            k++;
        }
}


void xyz_to_baselines(
        const double vlas[][3],
        int nants_vla,
        double ha_start,
        double ha_end,
        double ha_step,
        int ha_num,
        double dec_rad,
        double* dist_uvw,
        int uvw_num
)
{
    int i, j, k, basel_num;
    int jstart, jend;
    double ha_tmp;
    double* uvw_ha;
    double* basel_uvw;

    // uvw array for xyz conversion
    uvw_ha = (double*)malloc(nants_vla * 3 * sizeof(double));
    if (uvw_ha == NULL)
    {
        printf("Memory not allocated.\n");
        exit(-1);
    }

    // Number of baselines per snapshot
    basel_num = (int)(nants_vla * (nants_vla - 1) / 2);
    basel_uvw = (double*)malloc(basel_num * 3 * sizeof(double));
    if (basel_uvw == NULL)
    {
        printf("Memory not allocated.\n");
        exit(-1);
    }

    // Loop over snapshots (HA range)
    for (i = 0; i < ha_num; ++i)
    {
        ha_tmp = ha_start + ha_step * i;

        // xyz->uvw conversion for the current HA
        xyz_to_uvw(vlas, ha_tmp, dec_rad, uvw_ha, nants_vla);

        // Calculate the baselines for the current HA
        baselines(uvw_ha, basel_uvw, nants_vla);

        // Rewrite the baselines to the dist_uvw array containing all baselines
        // TODO: replace with memcpy
        jstart = 3 * i * basel_num;
        jend = 3 * (i + 1) * basel_num;
        k = 0;
        for (j = jstart; j < jend; j++)
        {
            dist_uvw[j] = basel_uvw[k];
            k++;
        }
    }

    free(uvw_ha);
    free(basel_uvw);
}


// void simulate_point(double *dist_uvw, sdp_Double2 *vis_sim, int uvw_num, double l, double m, double amp){
void simulate_point(
        double* dist_uvw,
        cufftDoubleComplex* vis_sim,
        int uvw_num,
        double l,
        double m,
        double amp
)
{
    double n, dot_product, dce;
    int i;

    n = sqrt(1.0 - l * l - m * m) - 1.0;

    for (i = 0; i < uvw_num; ++i)
    {
        dot_product = dist_uvw[3 * i] * l + dist_uvw[3 * i + 1] * m +
                dist_uvw[3 * i + 2] * n;
        dce = -2.0 * M_PI * dot_product;
        vis_sim[i].x += amp * cos(dce);
        vis_sim[i].y += amp * sin(dce);
        // printf("%d %f %f\n",i,creal(vis_sim[i]), cimag(vis_sim[i]));
    }
}


void find_minimax(sdp_Double2* arr, int n_arr, double* minimax)
{
    int i;
    minimax[0] = DBL_MAX;      // min creal
    minimax[1] = -(DBL_MAX - 1); // max creal
    minimax[2] = DBL_MAX;      // min cimag
    minimax[3] = -(DBL_MAX - 1); // max cimag
    for (i = 0; i < n_arr; ++i)
    {
        if (arr[i].x < minimax[0]) // min creal
            minimax[0] = arr[i].x;
        if (arr[i].x > minimax[1])  // max creal
            minimax[1] = arr[i].x;
        if (arr[i].y < minimax[2]) // min cimag
            minimax[2] = arr[i].y;
        if (arr[i].y > minimax[3])  // max cimag
            minimax[3] = arr[i].y;
    }
}


void sdp_grid_simulator_VLA(
        sdp_Mem* source_list,
        double ha_start,
        double ha_step,
        int ha_num,
        double dec_rad,
        double uvw_scale,
        sdp_Mem* grid_sim,
        sdp_Error* status
)
{
    int64_t i, j;
    int basel_num, uvw_num;
    int64_t grid_size, grid_centre;
    double ha_end;
    double* dist_uvw;
    cufftDoubleComplex* vis_sim;

    SDP_LOG_INFO("Inside sdp_grid_simulator_VLA");

    int64_t num_sources = sdp_mem_shape_dim(source_list, 0);
    SDP_LOG_INFO("The list includes %d sources", num_sources);

    double* est_sources = (double*) sdp_mem_data(source_list);
    for (size_t i = 0; i < num_sources; i++)
        //                            Amp                l                   m
        SDP_LOG_INFO("%f %f %f",
                est_sources[3 * i],
                est_sources[3 * i + 1],
                est_sources[3 * i + 2]
        );

    ha_end = ha_start + ha_step * (ha_num - 1);
    uvw_num = ha_num * nants_vla * (nants_vla - 1) / 2;
    SDP_LOG_INFO("Number of UVW points is %d", uvw_num);

    // Allocate double UVW array
    dist_uvw = (double*)malloc(uvw_num * 3 * sizeof(double));
    if (dist_uvw == NULL)
    {
        printf("Memory not allocated.\n");
        exit(-1);
    }

    // Allocate complex visibility array
    vis_sim = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex) * uvw_num);
    if (vis_sim == NULL)
    {
        printf("Memory not allocated.\n");
        exit(-1);
    }
    for (i = 0; i < uvw_num; ++i)
    {
        vis_sim[i].x = 0;
        vis_sim[i].y = 0;
    }

    // Calculate uvw points
    basel_num = (int)(nants_vla * (nants_vla - 1) / 2);
    xyz_to_baselines(vlas,
            nants_vla,
            ha_start,
            ha_end,
            ha_step,
            ha_num,
            dec_rad,
            dist_uvw,
            uvw_num
    );

    // Scale down uvw range, reduce uvw values in uvw_scale times
    SDP_LOG_INFO("Scaling down UVW range in %f times", uvw_scale);
    for (i = 0; i < uvw_num; i++)
    {
        dist_uvw[3 * i] /= uvw_scale;
        dist_uvw[3 * i + 1] /= uvw_scale;
        dist_uvw[3 * i + 2] /= uvw_scale;
    }

    // Model point sources contribution
    for (i = 0; i < num_sources; i++)
    {
        //                                                   l                m                 Amp
        simulate_point(dist_uvw,
                vis_sim,
                uvw_num,
                est_sources[3 * i + 1],
                est_sources[3 * i + 2],
                est_sources[3 * i]
        );
    }

    // Create the simulated grid
    grid_size = sdp_mem_shape_dim(grid_sim, 0);
    grid_centre = (int)(grid_size / 2 - 1);
    SDP_LOG_INFO("Grid size is %d, centre is %d", grid_size, grid_centre);

    // Assign zeros to the grid elements
    void* grid_sim_1 = (void*)sdp_mem_data(grid_sim);
    sdp_Double2* temp = (sdp_Double2*)grid_sim_1;

    for (i = 0; i < grid_size * grid_size; i++)
    {
        temp[i].x = 0.;
        temp[i].y = 0.;
    }

    // Fill the grid with vis_sim values

    int64_t iu_grid, iv_grid, iu_gridc, iv_gridc;
    j = 0;
    for (i = 0; i < uvw_num; ++i)
    {
        iu_grid = (int64_t)(dist_uvw[3 * i] + grid_centre);
        iv_grid = (int64_t)(dist_uvw[3 * i + 1] + grid_centre);

        iu_gridc = (int64_t)(grid_centre - dist_uvw[3 * i]);
        iv_gridc = (int64_t)(grid_centre - dist_uvw[3 * i + 1]);

        if ((iu_grid >= 0 && iu_grid < grid_size) &&
                (iv_grid >= 0 && iv_grid < grid_size))
        {
            temp[iu_grid * grid_size + iv_grid].x += vis_sim[i].x;
            temp[iu_grid * grid_size + iv_grid].y += vis_sim[i].y;

            temp[iu_gridc * grid_size + iv_gridc].x += vis_sim[i].x;
            temp[iu_gridc * grid_size + iv_gridc].y += -1.0 * vis_sim[i].y;
        }
        else
        {
            SDP_LOG_INFO("Index is outside the range, %d %d %d",
                    (int)i,
                    (int)iu_grid,
                    (int)iv_grid
            );
            j++;
        }
    }
    SDP_LOG_INFO("%d scaled vis points outside the range %d x %d",
            (int)j,
            (int)grid_size,
            (int)grid_size
    );

    SDP_LOG_INFO("Grid simulation is completed");

    free(dist_uvw);
    free(vis_sim);
}
